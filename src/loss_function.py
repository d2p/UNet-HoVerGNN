import torch
import torch.nn as nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Global one-hot Dice loss from HoVer-Net Eq. 5. Returns 1 - Dice."""

    def __init__(self, smooth: float = 1e-3):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes).float()
        if logits.dim() > 2:
            # [N, H, W, C] -> [N, C, H, W]
            one_hot = one_hot.permute(0, -1, *range(1, targets.ndim))
        intersection = (probs * one_hot).sum()
        denom = probs.sum() + one_hot.sum()
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice


class _NPBranchLoss(nn.Module):
    """Cross-entropy + Dice, both weight 1, for the NP and NC branches."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets) + self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Multi-class focal loss used by the original graph head."""

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class _GCBranchLoss(nn.Module):
    """Original graph-head loss: cross-entropy + Dice + focal, each weight 1.

    Pass graph_loss='gc' to HoverLoss to train with this instead of the
    HoVer-Net CE+Dice graph loss. Dice is the true 1 - Dice term.
    """

    def __init__(self, alpha=1, beta=1, gamma=1):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.alpha * F.cross_entropy(logits, targets)
            + self.beta * self.dice(logits, targets)
            + self.gamma * self.focal(logits, targets)
        )


def _sobel_x(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    return F.conv2d(image.unsqueeze(1), kernel.view(1, 1, 3, 3), padding=1).squeeze(1)


def _sobel_y(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return F.conv2d(image.unsqueeze(1), kernel.view(1, 1, 3, 3), padding=1).squeeze(1)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom.item() == 0:
        return pred.new_zeros(())
    return ((pred - target).square() * mask).sum() / denom


class _HVBranchLoss(nn.Module):
    """HoVer-Net horizontal/vertical loss: map MSE (weight 1) + gradient MSE (weight 2).

    The gradient term is the Sobel x-derivative of the horizontal map and the
    Sobel y-derivative of the vertical map, compared with the ground truth on
    nuclear pixels only.
    """

    def forward(self, logits, h_grads, v_grads, nuclear_mask):
        pred_h = logits[:, 0]
        pred_v = logits[:, 1]
        map_mse = F.mse_loss(pred_h, h_grads) + F.mse_loss(pred_v, v_grads)
        mask = nuclear_mask.to(dtype=pred_h.dtype)
        grad_mse = _masked_mse(_sobel_x(pred_h), _sobel_x(h_grads), mask)
        grad_mse = grad_mse + _masked_mse(_sobel_y(pred_v), _sobel_y(v_grads), mask)
        return map_mse + 2.0 * grad_mse


class HoverLoss(nn.Module):
    def __init__(self, graph_loss="ce_dice"):
        super().__init__()
        self.np_loss = _NPBranchLoss()
        self.hv_loss = _HVBranchLoss()
        self.nc_loss = _NPBranchLoss()
        if graph_loss == "ce_dice":
            self.gc_loss = _NPBranchLoss()
        elif graph_loss == "gc":
            self.gc_loss = _GCBranchLoss()
        else:
            raise ValueError("graph_loss must be 'ce_dice' or 'gc'")

    def forward(
        self,
        np_logits,
        np_targets,
        hv_logits,
        h_grads,
        v_grads,
        nc_logits,
        nc_targets,
        gc_logits=None,
        gc_targets=None,
        weights=(1, 1, 1, 1),
    ) -> torch.Tensor:
        loss_np = self.np_loss(np_logits, np_targets) * weights[0]
        loss_hv = self.hv_loss(hv_logits, h_grads, v_grads, np_targets) * weights[1]
        loss_nc = self.nc_loss(nc_logits, nc_targets) * weights[2]

        if gc_logits is not None and gc_targets is not None and gc_logits.numel() > 0:
            loss_gc = self.gc_loss(gc_logits, gc_targets) * weights[3]
        else:
            loss_gc = torch.zeros((), device=np_logits.device, dtype=loss_np.dtype)

        return loss_np + loss_hv + loss_nc + loss_gc
