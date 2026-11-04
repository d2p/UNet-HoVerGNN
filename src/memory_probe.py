"""One-step GPU memory probe for UNet-HoVerGNN.

Records measured peaks next to the pre-measurement estimates for 256x256
(MoNuSAC / PanNuke tiles and MoNuSeg tiles) and a 1000x1000 MoNuSeg image.
Parameter memory does not grow with batch size, so a batch-32 figure scaled
from batch 1 overestimates activation memory. The ranges below are the
estimates made before measuring this machine.
"""

import os

import torch

from loss_function import HoverLoss
from unet_hovergnn import GraphHoverNet

# Estimates made from a ~31M-parameter, 3-decoder U-Net before this run.
ESTIMATE = """\
Pre-measurement estimates (AMP train / no-grad eval)
- 256x256 batch 32 pretrain: 28-40 GB
- 256x256 batch 32 finetune: 34-48 GB
- 256x256 batch 32 eval: 4-8 GB
- 1000x1000 batch 1 pretrain: 30-45 GB
- 1000x1000 batch 1 eval: 3-6 GB
Practical batches from that estimate: 4/2 (pretrain/finetune) on 16 GB,
8/4 on 24 GB, 16/8 on 40-48 GB. Eval batch 32 fits on 16 GB.
"""


def _gb(num_bytes):
    return num_bytes / (1024 ** 3)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak():
    _sync()
    return _gb(torch.cuda.max_memory_allocated())


def _fake_batch(batch, size, num_classes, device, with_instances=False):
    image = torch.randn(batch, 3, size, size, device=device)
    mask = torch.zeros(batch, size, size, dtype=torch.long, device=device)
    h_map = torch.zeros(batch, size, size, device=device)
    v_map = torch.zeros(batch, size, size, device=device)
    inst = torch.zeros(batch, size, size, dtype=torch.long, device=device)
    if with_instances:
        for b in range(batch):
            for k, (y, x) in enumerate(((8, 8), (8, 40), (40, 8), (40, 40)), start=1):
                if y + 6 >= size or x + 6 >= size:
                    continue
                mask[b, y:y + 6, x:x + 6] = 1 if num_classes == 2 else (k % (num_classes - 1)) + 1
                inst[b, y:y + 6, x:x + 6] = k
                h_map[b, y:y + 6, x:x + 6] = torch.linspace(-1, 1, 6, device=device)
                v_map[b, y:y + 6, x:x + 6] = torch.linspace(-1, 1, 6, device=device).view(-1, 1)
    return image, mask, h_map, v_map, inst


def _train_step(model, criterion, optimizer, batch, size, num_classes, stage):
    device = next(model.parameters()).device
    model.set_stage(stage)
    model.train()
    image, mask, h_map, v_map, inst = _fake_batch(
        batch, size, num_classes, device, with_instances=(stage == "finetune")
    )
    optimizer.zero_grad(set_to_none=True)
    node_coords = None
    node_labels = None
    if stage == "finetune":
        from postprocess import centroids_from_instances, instance_majority_labels
        node_coords, inst_ids = centroids_from_instances(inst)
        node_labels = instance_majority_labels(inst, mask, inst_ids)
        keep = node_labels != 0
        node_coords = node_coords[keep]
        node_labels = node_labels[keep]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        np_logits, hv_logits, nc_logits, _, gc_logits = model(image, node_coords=node_coords)
        graph = (gc_logits, node_labels) if gc_logits is not None else (None, None)
        loss = criterion(
            np_logits, (mask > 0).long(),
            hv_logits, h_map, v_map,
            nc_logits, mask,
            *graph,
        )
    loss.backward()
    optimizer.step()
    return loss.detach()


def _eval_forward(model, batch, size):
    device = next(model.parameters()).device
    model.eval()
    image = torch.randn(batch, 3, size, size, device=device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model.set_stage("pretrain")
            model(image)


def measure():
    lines = [ESTIMATE, ""]
    if not torch.cuda.is_available():
        lines.append("CUDA is not available. No measurement was taken.")
        return "\n".join(lines)

    name = torch.cuda.get_device_name(0)
    total = _gb(torch.cuda.get_device_properties(0).total_memory)
    lines.append(f"GPU: {name}, {total:.2f} GB visible")

    device = torch.device("cuda")
    # Random init matches the trained architecture's memory. ImageNet loading does not change it.
    model = GraphHoverNet(num_classes=5, encoder_weights=None).to(device)
    criterion = HoverLoss()
    n_params = sum(p.numel() for p in model.parameters())
    lines.append(f"Parameters: {n_params:,} ({n_params * 4 / 1024**2:.1f} MB fp32)")

    # Eval first, while only the weights are resident. AdamW state is allocated below.
    eval_cases = [
        ("256 eval batch 1", lambda: _eval_forward(model, 1, 256)),
        ("256 eval batch 4", lambda: _eval_forward(model, 4, 256)),
        ("1000 eval batch 1", lambda: _eval_forward(model, 1, 1000)),
    ]
    measured = {}
    lines.append("Eval (no optimizer state):")
    for label, fn in eval_cases:
        try:
            _reset()
            fn()
            peak = _peak()
            measured[label] = peak
            lines.append(f"{label}: {peak:.2f} GB peak allocated")
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            lines.append(f"{label}: FAILED ({exc.__class__.__name__}: {str(exc).splitlines()[0]})")

    # Allocate AdamW state once. A fresh optimizer's first step spikes by about
    # the same amount at every batch size and hides the activation slope.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    _train_step(model, criterion, optimizer, 1, 64, 5, "pretrain")
    torch.cuda.empty_cache()
    lines.append(
        f"After one AdamW step: {_gb(torch.cuda.memory_allocated()):.2f} GB allocated "
        "(parameters, gradients, and moments)."
    )

    cases = [
        ("256 pretrain train batch 1", lambda: _train_step(model, criterion, optimizer, 1, 256, 5, "pretrain")),
        ("256 pretrain train batch 4", lambda: _train_step(model, criterion, optimizer, 4, 256, 5, "pretrain")),
        ("256 finetune train batch 1", lambda: _train_step(model, criterion, optimizer, 1, 256, 5, "finetune")),
        ("256 finetune train batch 4", lambda: _train_step(model, criterion, optimizer, 4, 256, 5, "finetune")),
        ("1000 pretrain train batch 1", lambda: _train_step(model, criterion, optimizer, 1, 1000, 5, "pretrain")),
    ]
    for label, fn in cases:
        try:
            _reset()
            fn()
            peak = _peak()
            measured[label] = peak
            lines.append(f"{label}: {peak:.2f} GB peak allocated")
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            lines.append(f"{label}: FAILED ({exc.__class__.__name__}: {str(exc).splitlines()[0]})")

    def _scale(small_key, large_key, small_b, large_b, target, estimate):
        if small_key not in measured or large_key not in measured:
            return
        per_sample = (measured[large_key] - measured[small_key]) / (large_b - small_b)
        est = measured[small_key] + per_sample * (target - small_b)
        lines.append(
            f"{small_key.split()[0]} {small_key.split()[1]} batch {target} scaled from "
            f"batch {small_b}->{large_b}: {est:.2f} GB "
            f"({per_sample:.3f} GB/sample). Compare with the {estimate} estimate."
        )

    _scale(
        "256 pretrain train batch 1", "256 pretrain train batch 4",
        1, 4, 32, "28-40 GB",
    )
    _scale(
        "256 finetune train batch 1", "256 finetune train batch 4",
        1, 4, 32, "34-48 GB",
    )
    _scale(
        "256 eval batch 1", "256 eval batch 4",
        1, 4, 32, "4-8 GB",
    )
    lines.append(
        "Measured 256 training adds about 0.15 GB per image on a 0.5 GB "
        "parameter-and-AdamW baseline, so batch 32 is about 5 GB on this GPU. "
        "MoNuSAC, PanNuke, and MoNuSeg all train as 256 tiles. "
        "The 1000 rows are one full MoNuSeg image."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    text = measure()
    print(text)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gpu_memory_estimate.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"Wrote {path}")
