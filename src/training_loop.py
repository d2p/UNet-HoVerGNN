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
from metrics import BinaryDiceMeter, HoverNetMetrics
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


def _pixel_class_scores(pred, target, num_classes):
    """Per-class pixel IoU and F1 for one batch. Class 0 is included."""
    ious = []
    f1s = []
    for class_id in range(num_classes):
        pred_c = pred == class_id
        true_c = target == class_id
        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & ~true_c).sum().float()
        fn = (~pred_c & true_c).sum().float()
        ious.append(tp / (tp + fp + fn + 1e-8))
        f1s.append(2.0 * tp / (2.0 * tp + fp + fn + 1e-8))
    return torch.stack(ious), torch.stack(f1s)


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

    post_process=True runs the HoVer watershed and reports instance AJI, DQ, SQ,
    PQ, and detection F1. The type map is the NC majority vote, replaced by the
    GNN class of each predicted instance when the graph stage is on.
    post_process=False scores the raw NC argmax with pixel IoU and pixel F1.
    Both paths report binary Dice of the NP head (nucleus versus background).
    For this two-class mask, Dice and foreground F1 are the same number.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    meter = HoverNetMetrics(num_classes=num_classes, radius=Config.MATCH_RADIUS) if post_process else None
    dice_meter = BinaryDiceMeter()
    total_pixel_iou = 0.0
    total_pixel_f1 = 0.0
    total_pixel_f1_per_class = torch.zeros(num_classes, dtype=torch.float32, device=device)
    use_amp = device.type == "cuda"

    loop = tqdm(dataloader, desc="Validation", leave=True)
    for batch in loop:
        image, mask, h_grads, v_grads, inst = _to_device(batch, device)
        nc_targets = mask
        np_targets = (mask > 0).long()
        # Instance ids are only needed when the watershed scores are being computed.
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
            # Instance ids from the watershed. GNN classes replace the NC vote per instance.
            pred_inst, pred_type = postprocess_hovernet_output(
                np_logits, hv_logits, nc_logits, device, nr_types=num_classes
            )
            if model.use_graph:
                pred_coords, pred_ids = centroids_from_instances(pred_inst)
                _, pred_logits = model.classify_nodes(image, pred_coords)
                pred_type = apply_gnn_instance_labels(pred_inst, pred_ids, pred_logits, pred_type)

            for b in range(image.shape[0]):
                meter.update(
                    gt_inst[b].detach().cpu().numpy(),
                    pred_inst[b].detach().cpu().numpy(),
                    mask[b].detach().cpu().numpy(),
                    pred_type[b].detach().cpu().numpy(),
                )
        else:
            # Raw class map. These are pixel scores, kept apart from AJI and PQ.
            nc_pred = torch.argmax(nc_logits, dim=1)
            iou_per_class, f1_per_class = _pixel_class_scores(nc_pred, nc_targets, num_classes)
            total_pixel_iou += iou_per_class[1:].mean().item()
            total_pixel_f1 += f1_per_class[1:].mean().item()
            total_pixel_f1_per_class += f1_per_class

        total_loss += loss.item()
        count += 1
        loop.set_postfix(loss=total_loss / max(count, 1))

    dice = dice_meter.compute()
    if post_process:
        scores = meter.compute()
        scores["loss"] = total_loss / max(count, 1)
        scores["dice"] = dice
        scores["post_process"] = True
        return scores

    class_f1 = (total_pixel_f1_per_class / max(count, 1)).tolist()
    return {
        "loss": total_loss / max(count, 1),
        "pixel_iou": total_pixel_iou / max(count, 1),
        "pixel_f1": total_pixel_f1 / max(count, 1),
        "pixel_f1_per_class": class_f1[1:],
        "dice": dice,
        "post_process": False,
    }


def train(model, criterion, train_loader, val_loader, device,
          epochs=20, lr=1e-4, num_classes=5, patience=5, post_process=True):

    def train_loop(stage_name, use_graph, start_epoch=1, lr=lr):
        nonlocal best_val_score
        nonlocal epochs_no_improve

        best_val_score = -float("inf")
        epochs_no_improve = 0

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
            val_dices.append(val_scores["dice"])
            if post_process:
                val_ajis.append(val_scores["aji"])
                val_dqs.append(val_scores["dq"])
                val_sqs.append(val_scores["sq"])
                val_pqs.append(val_scores["pq"])
                val_fds.append(val_scores["f1_d"])
                monitor = val_scores["pq"]
                per_class = val_scores["f1_per_class"]
            else:
                val_pixel_ious.append(val_scores["pixel_iou"])
                val_pixel_f1s.append(val_scores["pixel_f1"])
                monitor = val_scores["pixel_f1"]
                per_class = val_scores["pixel_f1_per_class"]

            print(f"Train Loss: {train_loss:.4f}")
            if post_process:
                print(
                    f"Val   Loss: {val_scores['loss']:.4f} | AJI: {val_scores['aji']:.4f} | "
                    f"DQ: {val_scores['dq']:.4f} | SQ: {val_scores['sq']:.4f} | "
                    f"PQ: {val_scores['pq']:.4f} | F1_d: {val_scores['f1_d']:.4f} | "
                    f"Dice: {val_scores['dice']:.4f}"
                )
                print("Val detection F1 per class: " + " | ".join(
                    f"Class {i + 1}: {score:.4f}" for i, score in enumerate(per_class)
                ))
            else:
                print(
                    f"Val   Loss: {val_scores['loss']:.4f} | pixel IoU: {val_scores['pixel_iou']:.4f} | "
                    f"pixel F1: {val_scores['pixel_f1']:.4f} | Dice: {val_scores['dice']:.4f}"
                )
                print("Val pixel F1 per class: " + " | ".join(
                    f"Class {i + 1}: {score:.4f}" for i, score in enumerate(per_class)
                ))

            if monitor > best_val_score:
                best_val_score = monitor
                epochs_no_improve = 0
                best_model_path = os.path.join(Config.OUTPUT_PATH, f"best_hovernet_{stage_name}.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model (based on {monitor_name}) to {best_model_path}.")
            else:
                epochs_no_improve += 1
                print(f"No improvement in {monitor_name} for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement in {monitor_name}.")
                break

        return current_epoch + 1

    train_losses, val_losses = [], []
    val_ajis, val_dqs, val_sqs, val_pqs, val_fds = [], [], [], [], []
    val_dices = []
    val_pixel_ious, val_pixel_f1s = [], []
    monitor_name = "PQ" if post_process else "pixel F1"

    best_val_score = -float("inf")
    epochs_no_improve = 0

    next_epoch = train_loop(stage_name="pretrain", use_graph=False, start_epoch=1)
    next_epoch = train_loop(stage_name="finetune", use_graph=True, start_epoch=next_epoch, lr=1e-2)

    print(f"\nTraining complete. Best Val {monitor_name}: {best_val_score:.4f}")

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

    if post_process:
        plt.subplot(1, 4, 2)
        plt.plot(epochs_range, val_ajis, label="Val AJI")
        plt.xlabel("Epoch")
        plt.ylabel("AJI")
        plt.title("AJI")
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
            "Val DQ": val_dqs,
            "Val SQ": val_sqs,
            "Val PQ": val_pqs,
            "Val F1_d": val_fds,
            "Val Dice": val_dices,
        }
    else:
        plt.subplot(1, 4, 2)
        plt.plot(epochs_range, val_pixel_ious, label="Val pixel IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Pixel IoU")
        plt.legend()

        plt.subplot(1, 4, 3)
        plt.plot(epochs_range, val_pixel_f1s, label="Val pixel F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Pixel F1")
        plt.legend()
        plt.subplot(1, 4, 4)
        plt.plot(epochs_range, val_dices, label="Val Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Binary Dice")
        plt.legend()
        metric_columns = {
            "Val pixel IoU": val_pixel_ious,
            "Val pixel F1": val_pixel_f1s,
            "Val Dice": val_dices,
        }

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
    if scores["post_process"]:
        print(
            f"\nFinal Test [{stage_name}] | Loss: {scores['loss']:.4f} | "
            f"AJI: {scores['aji']:.4f} | DQ: {scores['dq']:.4f} | "
            f"SQ: {scores['sq']:.4f} | PQ: {scores['pq']:.4f} | F1_d: {scores['f1_d']:.4f} | "
            f"Dice: {scores['dice']:.4f}"
        )
        print("Per-class detection F1:", scores["f1_per_class"])
        metrics = {
            "Loss": [scores["loss"]],
            "AJI": [scores["aji"]],
            "DQ": [scores["dq"]],
            "SQ": [scores["sq"]],
            "PQ": [scores["pq"]],
            "F1_d": [scores["f1_d"]],
            "Dice": [scores["dice"]],
        }
        per_class = scores["f1_per_class"]
    else:
        print(
            f"\nFinal Test [{stage_name}] | Loss: {scores['loss']:.4f} | "
            f"pixel IoU: {scores['pixel_iou']:.4f} | pixel F1: {scores['pixel_f1']:.4f} | "
            f"Dice: {scores['dice']:.4f}"
        )
        print("Per-class pixel F1:", scores["pixel_f1_per_class"])
        metrics = {
            "Loss": [scores["loss"]],
            "pixel_IoU": [scores["pixel_iou"]],
            "pixel_F1": [scores["pixel_f1"]],
            "Dice": [scores["dice"]],
        }
        per_class = scores["pixel_f1_per_class"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.OUTPUT_PATH, f"final_eval_{stage_name}_{timestamp}.csv")
    for i, f1 in enumerate(per_class, start=1):
        metrics[f"F1_Class_{i}"] = [f1]
    pd.DataFrame(metrics).to_csv(results_file, index=False)
    print(f"Saved final evaluation metrics to {results_file}")
    return scores
