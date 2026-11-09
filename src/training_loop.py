from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast
from postprocess import (
    apply_gnn_instance_labels,
    centroids_from_instances,
    instance_majority_labels,
    postprocess_hovernet_output,
    resolve_gt_instances,
)
import os
from config import Config
from metrics import BinaryDiceMeter, HoverNetMetrics, PixelComparisonMeter
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def _to_device(batch, device):
    image, mask, h_grads, v_grads, inst = batch
    return (
        image.to(device),
        mask.to(device).long(),
        h_grads.to(device),
        v_grads.to(device),
        inst.to(device),
    )


def _gt_graph_nodes(model, mask, h_grads, v_grads, inst):
    """One training node per ground-truth nucleus. Unused while the graph is off."""
    if not model.use_graph:
        return None, None
    gt_inst = resolve_gt_instances(mask, h_grads, v_grads, inst)
    coords, inst_ids = centroids_from_instances(gt_inst)
    labels = instance_majority_labels(gt_inst, mask, inst_ids)
    keep = labels != 0
    if not bool(keep.any()):
        return None, None
    return coords[keep], labels[keep]


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    count = 0
    use_amp = device.type == "cuda"

    loop = tqdm(dataloader, desc="Training", leave=True)
    for batch in loop:
        image, mask, h_grads, v_grads, inst = _to_device(batch, device)
        nc_targets = mask
        np_targets = (mask > 0).long()
        node_coords, node_labels = _gt_graph_nodes(model, mask, h_grads, v_grads, inst)

        optimizer.zero_grad()
        with autocast(device.type, enabled=use_amp):
            np_logits, hv_logits, nc_logits, _, gc_logits = model(image, node_coords=node_coords)
            if gc_logits is not None and node_labels is not None:
                graph_inputs = (gc_logits, node_labels)
            else:
                graph_inputs = (None, None)
            loss = criterion(
                np_logits, np_targets,
                hv_logits, h_grads, v_grads,
                nc_logits, nc_targets,
                *graph_inputs,
            )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        count += 1
        loop.set_postfix(loss=total_loss / count)

    return total_loss / max(count, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes=5, post_process=True):
    """Score one split.

    Comparison AJI, F1, and PQ are pixel scores on the semantic class map.
    AJI is the mean foreground pixel IoU, F1 is the mean foreground pixel F1,
    and PQ is the mean over classes of their harmonic mean. With post-processing
    on, that class map is the instance type map, GNN labels included. Otherwise
    it is the raw NC argmax. Binary Dice of the NP head is always reported.
    Post-processed runs also keep HoVer-Net instance scores under inst_* keys.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    meter = HoverNetMetrics(num_classes=num_classes, radius=Config.MATCH_RADIUS) if post_process else None
    pixel_meter = PixelComparisonMeter(num_classes)
    dice_meter = BinaryDiceMeter()
    use_amp = device.type == "cuda"

    loop = tqdm(dataloader, desc="Validation", leave=True)
    for batch in loop:
        image, mask, h_grads, v_grads, inst = _to_device(batch, device)
        nc_targets = mask
        np_targets = (mask > 0).long()
        # Instance ids are only needed when the post-processed scores are being computed.
        gt_inst = resolve_gt_instances(mask, h_grads, v_grads, inst) if post_process else None
        node_coords, node_labels = _gt_graph_nodes(model, mask, h_grads, v_grads, inst)

        with autocast(device.type, enabled=use_amp):
            np_logits, hv_logits, nc_logits, _, gc_logits = model(image, node_coords=node_coords)
            if gc_logits is not None and node_labels is not None:
                graph_inputs = (gc_logits, node_labels)
            else:
                graph_inputs = (None, None)
            loss = criterion(
                np_logits, np_targets,
                hv_logits, h_grads, v_grads,
                nc_logits, nc_targets,
                *graph_inputs,
            )

        # Binary segmentation of the nuclear head. Class labels are collapsed
        # to nucleus versus background, which is the NP training target.
        pred_fg = torch.argmax(np_logits, dim=1) > 0
        dice_meter.update(pred_fg.detach().cpu().numpy(), (mask > 0).detach().cpu().numpy())

        if post_process:
            # Instance ids from post-processing. GNN classes replace the NC vote per instance.
            pred_inst, pred_type = postprocess_hovernet_output(
                np_logits, hv_logits, nc_logits, device, nr_types=num_classes
            )
            if model.use_graph:
                pred_coords, pred_ids = centroids_from_instances(pred_inst)
                _, pred_logits = model.classify_nodes(image, pred_coords)
                pred_type = apply_gnn_instance_labels(pred_inst, pred_ids, pred_logits, pred_type)

            pred_type_np = pred_type.detach().cpu().numpy()
            gt_np = nc_targets.detach().cpu().numpy()
            pixel_meter.update(pred_type_np, gt_np)
            for b in range(image.shape[0]):
                meter.update(
                    gt_inst[b].detach().cpu().numpy(),
                    pred_inst[b].detach().cpu().numpy(),
                    gt_np[b],
                    pred_type_np[b],
                )
        else:
            pixel_meter.update(
                torch.argmax(nc_logits, dim=1).detach().cpu().numpy(),
                nc_targets.detach().cpu().numpy(),
            )

        total_loss += loss.item()
        count += 1
        loop.set_postfix(loss=total_loss / max(count, 1))

    pixel_scores = pixel_meter.compute()
    scores = {
        "loss": total_loss / max(count, 1),
        "dice": dice_meter.compute(),
        "aji": pixel_scores["aji"],
        "f1": pixel_scores["f1"],
        "pq": pixel_scores["pq"],
        "f1_per_class": pixel_scores["f1_per_class"],
        "post_process": post_process,
    }
    if post_process:
        inst_scores = meter.compute()
        scores["inst_aji"] = inst_scores["aji"]
        scores["inst_dq"] = inst_scores["dq"]
        scores["inst_sq"] = inst_scores["sq"]
        scores["inst_pq"] = inst_scores["pq"]
        scores["f1_d"] = inst_scores["f1_d"]
    return scores


def train(model, criterion, train_loader, val_loader, device,
          epochs=20, lr=1e-4, num_classes=5, post_process=True):
    """Train pretrain then finetune for the full epoch budget.

    The saved checkpoint is the best validation PQ, the harmonic mean of
    pixel IoU and pixel F1. Training does not stop early: stopping on the
    instance score was cutting off later segmentation gains.
    """

    def train_loop(stage_name, use_graph, start_epoch=1, lr=lr):
        nonlocal best_val_score

        best_val_score = -float("inf")

        print(f"\n=== Starting {stage_name} phase ===")
        model.set_stage("pretrain" if not use_graph else "finetune")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = GradScaler(device.type, enabled=device.type == "cuda")

        for epoch in range(epochs):
            current_epoch = start_epoch + epoch
            print(f"\nEpoch {current_epoch}/{start_epoch + epochs - 1} [{stage_name}]")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_scores = validate(
                model, val_loader, criterion, device, num_classes, post_process=post_process
            )

            train_losses.append(train_loss)
            val_losses.append(val_scores["loss"])
            val_ajis.append(val_scores["aji"])
            val_f1s.append(val_scores["f1"])
            val_pqs.append(val_scores["pq"])
            val_dices.append(val_scores["dice"])
            if post_process:
                val_inst_ajis.append(val_scores["inst_aji"])
                val_inst_pqs.append(val_scores["inst_pq"])
                val_fds.append(val_scores["f1_d"])

            print(f"Train Loss: {train_loss:.4f}")
            print(
                f"Val   Loss: {val_scores['loss']:.4f} | AJI: {val_scores['aji']:.4f} | "
                f"F1: {val_scores['f1']:.4f} | PQ: {val_scores['pq']:.4f} | "
                f"Dice: {val_scores['dice']:.4f}"
            )
            print("Val F1 per class: " + " | ".join(
                f"Class {i + 1}: {score:.4f}" for i, score in enumerate(val_scores["f1_per_class"])
            ))
            if post_process:
                print(
                    f"Instance AJI: {val_scores['inst_aji']:.4f} | DQ: {val_scores['inst_dq']:.4f} | "
                    f"SQ: {val_scores['inst_sq']:.4f} | PQ: {val_scores['inst_pq']:.4f} | "
                    f"F1_d: {val_scores['f1_d']:.4f}"
                )

            if val_scores["pq"] > best_val_score:
                best_val_score = val_scores["pq"]
                best_model_path = os.path.join(Config.OUTPUT_PATH, f"best_hovernet_{stage_name}.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model (based on PQ) to {best_model_path}.")

        return current_epoch + 1

    train_losses, val_losses = [], []
    val_ajis, val_f1s, val_pqs, val_dices = [], [], [], []
    val_inst_ajis, val_inst_pqs, val_fds = [], [], []

    best_val_score = -float("inf")

    next_epoch = train_loop(stage_name="pretrain", use_graph=False, start_epoch=1)
    next_epoch = train_loop(stage_name="finetune", use_graph=True, start_epoch=next_epoch, lr=1e-2)

    print(f"\nTraining complete. Best Val PQ: {best_val_score:.4f}")

    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs_range, val_ajis, label="Val AJI")
    plt.plot(epochs_range, val_f1s, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Pixel IoU and F1")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs_range, val_pqs, label="Val PQ")
    plt.xlabel("Epoch")
    plt.ylabel("PQ")
    plt.title("PQ")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs_range, val_dices, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Binary Dice")
    plt.legend()
    metric_columns = {
        "Val AJI": val_ajis,
        "Val F1": val_f1s,
        "Val PQ": val_pqs,
        "Val Dice": val_dices,
    }
    if post_process:
        metric_columns.update({
            "Val inst AJI": val_inst_ajis,
            "Val inst PQ": val_inst_pqs,
            "Val F1_d": val_fds,
        })

    plt.tight_layout()
    plot_path = os.path.join(Config.OUTPUT_PATH, f"training_metrics_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training metrics plot to {plot_path}")

    metrics_df = pd.DataFrame({
        "Epoch": list(epochs_range),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        **metric_columns,
    })
    csv_path = os.path.join(Config.OUTPUT_PATH, f"training_metrics_{timestamp}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved training metrics to {csv_path}")


@torch.no_grad()
def final_evaluate(model, test_loader, criterion, device, stage_name, num_classes=6, post_process=True):
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    model_path = os.path.join(Config.OUTPUT_PATH, f"best_hovernet_{stage_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.set_stage(stage_name)

    scores = validate(
        model, test_loader, criterion, device, num_classes, post_process=post_process
    )
    print(
        f"\nFinal Test [{stage_name}] | Loss: {scores['loss']:.4f} | "
        f"AJI: {scores['aji']:.4f} | F1: {scores['f1']:.4f} | "
        f"PQ: {scores['pq']:.4f} | Dice: {scores['dice']:.4f}"
    )
    print("Per-class pixel F1:", scores["f1_per_class"])
    metrics = {
        "Loss": [scores["loss"]],
        "AJI": [scores["aji"]],
        "F1": [scores["f1"]],
        "PQ": [scores["pq"]],
        "Dice": [scores["dice"]],
    }
    if scores["post_process"]:
        print(
            f"Instance AJI: {scores['inst_aji']:.4f} | DQ: {scores['inst_dq']:.4f} | "
            f"SQ: {scores['inst_sq']:.4f} | PQ: {scores['inst_pq']:.4f} | "
            f"F1_d: {scores['f1_d']:.4f}"
        )
        metrics.update({
            "inst_AJI": [scores["inst_aji"]],
            "inst_DQ": [scores["inst_dq"]],
            "inst_SQ": [scores["inst_sq"]],
            "inst_PQ": [scores["inst_pq"]],
            "F1_d": [scores["f1_d"]],
        })
    per_class = scores["f1_per_class"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.OUTPUT_PATH, f"final_eval_{stage_name}_{timestamp}.csv")
    for i, f1 in enumerate(per_class, start=1):
        metrics[f"F1_Class_{i}"] = [f1]
    pd.DataFrame(metrics).to_csv(results_file, index=False)
    print(f"Saved final evaluation metrics to {results_file}")
    return scores
