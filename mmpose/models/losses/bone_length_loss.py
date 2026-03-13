import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple
from mmpose.registry import MODELS

EPS = 1e-9
@MODELS.register_module()
class BoneVectorSimCCLoss(nn.Module):
    """Bone vector consistency loss for SimCC outputs.

    Uses softargmax on x/y probability vectors to get differentiable coordinates,
    then penalizes L1 difference between predicted and GT bone vectors.

    Inputs:
      pred_simcc: (pred_x, pred_y), each [N, K, L]
      gt_simcc:   (gt_x, gt_y),     each [N, K, L]
      target_weight: [N, K]

    Assumption:
      target_weight[n, k] > 0 means joint k is valid for sample n.
    """

    def __init__(
        self,
        beta: float = 10.0,
        label_beta: float = 10.0,
        loss_weight: float = 0.05,
        use_target_weight: bool = True,
        bone_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        simcc_split_ratio: float = 2.0,
    ):
        super().__init__()
        self.beta = float(beta)
        self.label_beta = float(label_beta)
        self.loss_weight = float(loss_weight)
        self.use_target_weight = bool(use_target_weight)
        self.simcc_split_ratio = float(simcc_split_ratio)

        # COCO-style limb edges
        if bone_pairs is None:
            bone_pairs = [
                (5, 7), (7, 9),      # left arm
                (6, 8), (8, 10),     # right arm
                (11, 13), (13, 15),  # left leg
                (12, 14), (14, 16),  # right leg
                (5, 6),              # shoulders
                (11, 12),            # hips
                (5, 11), (6, 12),    # torso links
            ]
        self.bone_pairs = list(bone_pairs)

    def _to_prob_pred(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits * self.beta, dim=-1)

    def _to_prob_gt(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert GT label vectors to probabilities.
        - valid nonzero rows -> softmax(labels * beta)
        - zero rows -> stay zero
        labels: [N, K, L]
        returns: [N, K, L]
        """
        beta = float(self.beta)
        s = labels.sum(dim=-1, keepdim=True)   # [N, K, 1]
        valid = s.squeeze(-1) > 0              # [N, K]

        out = torch.zeros_like(labels)
        if valid.any():
            out[valid] = F.softmax(labels[valid] * beta, dim=-1)
        return out

    def _softargmax_1d(self, p: torch.Tensor) -> torch.Tensor:
        L = p.shape[-1]
        coords = torch.arange(L, device=p.device, dtype=p.dtype).view(1, 1, L)
        return (p * coords).sum(dim=-1)

    def forward(
        self,
        pred_simcc: Tuple[torch.Tensor, torch.Tensor],
        gt_simcc: Tuple[torch.Tensor, torch.Tensor],
        target_weight: torch.Tensor,
        epoch=None
    ) -> torch.Tensor:
        pred_x, pred_y = pred_simcc   # [N, K, Lx], [N, K, Ly]
        gt_x, gt_y = gt_simcc         # [N, K, Lx], [N, K, Ly]

        px = self._to_prob_pred(pred_x)
        py = self._to_prob_pred(pred_y)
        gx = self._to_prob_gt(gt_x)
        gy = self._to_prob_gt(gt_y)

        # differentiable coordinates in bin units
        pred_coord_x = self._softargmax_1d(px) / self.simcc_split_ratio  # [N, K]
        pred_coord_y = self._softargmax_1d(py) / self.simcc_split_ratio  # [N, K]
        gt_coord_x   = self._softargmax_1d(gx) / self.simcc_split_ratio  # [N, K]
        gt_coord_y   = self._softargmax_1d(gy) / self.simcc_split_ratio  # [N, K]

        if self.use_target_weight:
            w = target_weight.float()  # [N, K]
        else:
            w = torch.ones_like(target_weight, dtype=pred_x.dtype)

        total_loss = pred_x.new_tensor(0.0)
        total_count = pred_x.new_tensor(0.0)

        for i, j in self.bone_pairs:
            # edge valid only if both joints valid
            edge_valid = (w[:, i] > 0) & (w[:, j] > 0)  # [N]

            if not edge_valid.any():
                continue

            pred_vx = pred_coord_x[:, i] - pred_coord_x[:, j]  # [N]
            pred_vy = pred_coord_y[:, i] - pred_coord_y[:, j]
            gt_vx   = gt_coord_x[:, i]   - gt_coord_x[:, j]
            gt_vy   = gt_coord_y[:, i]   - gt_coord_y[:, j]

            edge_loss = (pred_vx - gt_vx).abs() + (pred_vy - gt_vy).abs()  # [N]
            edge_loss = edge_loss * edge_valid.float()

            total_loss = total_loss + edge_loss.sum()
            total_count = total_count + edge_valid.float().sum()

        if total_count.item() == 0:
            return pred_x.new_tensor(0.0)

        return self.loss_weight * (total_loss / (total_count + EPS))