"""
Shared loss functions for crack segmentation.
All methods import build_criterion from here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.float().view(-1)
        target = target.float().view(-1)
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), target, reduction='none')
        pt      = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.float().view(-1)
        target = target.float().view(-1)
        inter  = (pred * target).sum()
        return 1 - (2. * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def _boundary(self, x: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(x, self.kernel_size, 1, self.pad)
        eroded  = -F.max_pool2d(-x, self.kernel_size, 1, self.pad)
        return (dilated - eroded).clamp(0, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p  = self._boundary(pred.float()).clamp(1e-7, 1 - 1e-7)
        gt = self._boundary(target.float())
        with torch.amp.autocast('cuda', enabled=False):
            return F.binary_cross_entropy(p, gt)


class CombinedLoss(nn.Module):
    """L = w_f·L_focal + w_d·L_dice + w_b·L_boundary"""

    def __init__(
        self,
        focal_weight: float    = 1.0,
        dice_weight: float     = 1.0,
        boundary_weight: float = 1.0,
        focal_alpha: float     = 0.85,
        focal_gamma: float     = 2.0,
        boundary_kernel: int   = 3,
        label_smoothing: float = 0.0,
        from_logits: bool      = False,
    ):
        super().__init__()
        self.focal_w        = focal_weight
        self.dice_w         = dice_weight
        self.boundary_w     = boundary_weight
        self.label_smoothing = label_smoothing
        self.from_logits    = from_logits

        self.focal    = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice     = DiceLoss()
        self.boundary = BoundaryLoss(kernel_size=boundary_kernel) if boundary_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p  = torch.sigmoid(pred.float()) if self.from_logits else pred.float()
        gt = target.float()

        if self.label_smoothing > 0:
            gt_smooth = gt * (1 - self.label_smoothing) + self.label_smoothing / 2
        else:
            gt_smooth = gt

        loss = self.focal_w * self.focal(p, gt_smooth) + self.dice_w * self.dice(p, gt)

        if self.boundary is not None and self.boundary_w > 0:
            loss = loss + self.boundary_w * self.boundary(p, gt)

        return loss


def build_criterion(config: dict) -> nn.Module:
    lc = config.get('loss', {})
    return CombinedLoss(
        focal_weight    = lc.get('focal_weight',    1.0),
        dice_weight     = lc.get('dice_weight',     1.0),
        boundary_weight = lc.get('boundary_weight', 1.0),
        focal_alpha     = lc.get('focal_alpha',     0.85),
        focal_gamma     = lc.get('focal_gamma',     2.0),
        boundary_kernel = lc.get('boundary_kernel', 3),
        label_smoothing = lc.get('label_smoothing', 0.0),
        from_logits     = config.get('model', {}).get('return_logits', False),
    )
