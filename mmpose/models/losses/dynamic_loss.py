# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS

EPS = 1e-9


def _renorm(p: torch.Tensor) -> torch.Tensor:
    # p: [..., L], non-negative
    return p / (p.sum(dim=-1, keepdim=True) + EPS)


@MODELS.register_module()
class DynamicStructuralSimCCLoss(nn.Module):
    """Dynamic Structural Loss for SimCC (RTMPose/RTMCC).

    This implements the paper idea "H_i^s = H_i + sum_{j in N(i)} H_j",
    but for SimCC by operating on 1D distributions along x and y.

    Inputs follow KLDiscretLoss style:
      pred_simcc: (pred_x, pred_y) each [N, K, L]
      gt_simcc:   (gt_x, gt_y)     each [N, K, L]
      target_weight: [N, K]

    Notes:
    - We sum *probabilities*, not logits.
    - We renormalize after summing.
    - We mask by target_weight for both the loss term and (optionally) neighbor contribution.
    """

    def __init__(
        self,
        beta: float = 10.0,
        label_softmax: bool = True,
        label_beta: float = 10.0,
        use_target_weight: bool = True,
        loss_weight: float = 1.0,
        # dynamic schedule
        warmup_epochs: int = 0,
        ramp_epochs: int = 50,
        schedule: str = "linear",  # "linear" or "step"
        # graph definition
        limb_joint_indices: Optional[Sequence[int]] = None,
        neighbor_map: Optional[Dict[int, Sequence[int]]] = None,
        # behavior knobs
        include_only_valid_neighbors: bool = True,
        # Optional per-joint masking similar to KLDiscretLoss
        mask: Optional[Sequence[int]] = None,
        mask_weight: float = 1.0,
    ):
        super().__init__()
        self.beta = float(beta)
        self.label_softmax = bool(label_softmax)
        self.label_beta = float(label_beta)
        self.use_target_weight = bool(use_target_weight)
        self.loss_weight = float(loss_weight)

        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.schedule = str(schedule).lower()

        self.include_only_valid_neighbors = bool(include_only_valid_neighbors)

        self.mask = list(mask) if mask is not None else None
        self.mask_weight = float(mask_weight)

        # Default: COCO-17 limb chains (indices are COCO standard)
        # Arms:  LW(9)-LE(7)-LS(5)-RS(6)-RE(8)-RW(10)
        # Legs:  LA(15)-LK(13)-LH(11)-RH(12)-RK(14)-RA(16)
        if limb_joint_indices is None:
            limb_joint_indices = [9, 7, 5, 6, 8, 10, 15, 13, 11, 12, 14, 16]
        if neighbor_map is None:
            neighbor_map = {
                9: [7],
                7: [9, 5],
                5: [7, 6],
                6: [5, 8],
                8: [6, 10],
                10: [8],
                15: [13],
                13: [15, 11],
                11: [13, 12],
                12: [11, 14],
                14: [12, 16],
                16: [14],
            }

        self.limb_joint_indices = list(limb_joint_indices)
        self.neighbor_map = {int(k): list(v) for k, v in neighbor_map.items()}

        self.kl = nn.KLDivLoss(reduction="none")

    def _alpha(self, epoch: Optional[int]) -> float:
        """Dynamic weight multiplier in [0, 1] (then multiplied by loss_weight)."""
        if epoch is None:
            # If you don't pass epoch, just apply full weight.
            return 1.0

        e = int(epoch)
        if e < self.warmup_epochs:
            return 0.0

        if self.schedule == "step":
            return 1.0

        # linear ramp after warmup
        if self.ramp_epochs <= 0:
            return 1.0

        t = (e - self.warmup_epochs) / float(self.ramp_epochs)
        return float(max(0.0, min(1.0, t)))

    def _to_prob_pred(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [N, K, L]
        return F.softmax(logits * self.beta, dim=-1)

    def _to_prob_gt(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: [N, K, L] either already probs or logits-like soft labels
        if self.label_softmax:
            return F.softmax(labels * self.label_beta, dim=-1)
        # assume already non-negative + normalized-ish; renorm to be safe
        labels = labels.clamp_min(0)
        return _renorm(labels)

    def _struct_sum(
        self,
        p: torch.Tensor,              # [N, K, L] probabilities
        w: torch.Tensor,              # [N, K] weights
    ) -> torch.Tensor:
        """Return structure probabilities ps: [N, K, L] but only meaningful on limb joints."""
        N, K, L = p.shape
        ps = torch.zeros_like(p)

        for i in self.limb_joint_indices:
            # base
            si = p[:, i, :]  # [N, L]

            # add neighbors
            for j in self.neighbor_map.get(i, []):
                if self.include_only_valid_neighbors:
                    # per-sample mask [N, 1]
                    mj = (w[:, j] > 0).float().unsqueeze(-1)
                    si = si + p[:, j, :] * mj
                else:
                    si = si + p[:, j, :]

            ps[:, i, :] = _renorm(si)

        return ps

    def forward(
        self,
        pred_simcc: Tuple[torch.Tensor, torch.Tensor],
        gt_simcc: Tuple[torch.Tensor, torch.Tensor],
        target_weight: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        pred_x, pred_y = pred_simcc  # [N, K, Lx], [N, K, Ly]
        gt_x, gt_y = gt_simcc        # [N, K, Lx], [N, K, Ly]

        N, K, _ = pred_x.shape

        if self.use_target_weight:
            w = target_weight  # [N, K]
        else:
            w = torch.ones((N, K), device=pred_x.device, dtype=pred_x.dtype)

        # prob space
        px = self._to_prob_pred(pred_x)
        py = self._to_prob_pred(pred_y)
        gx = self._to_prob_gt(gt_x)
        gy = self._to_prob_gt(gt_y)

        # structure sums
        psx = self._struct_sum(px, w)
        psy = self._struct_sum(py, w)
        gsx = self._struct_sum(gx, w)
        gsy = self._struct_sum(gy, w)

        # compute KL in the same direction as KLDiscretLoss: KL(target || pred)
        # KLDivLoss expects input = log-prob, target = prob
        log_psx = (psx + EPS).log()
        log_psy = (psy + EPS).log()

        # per (N,K,L) -> per (N,K)
        loss_x = self.kl(log_psx, gsx).mean(dim=-1)
        loss_y = self.kl(log_psy, gsy).mean(dim=-1)

        loss_nk = loss_x + loss_y  # [N, K]

        # only keep limb joints
        limb_mask = torch.zeros((K,), device=pred_x.device, dtype=torch.bool)
        limb_mask[self.limb_joint_indices] = True
        loss_nk = loss_nk[:, limb_mask]
        w_limb = w[:, limb_mask]

        # apply target weights
        loss_nk = loss_nk * w_limb

        # optional extra masking like KLDiscretLoss
        if self.mask is not None:
            # mask indices refer to original K indexing
            # If any of them are limb joints, apply mask_weight
            for idx in self.mask:
                if 0 <= idx < K and limb_mask[idx]:
                    # find its column in the filtered limb tensor
                    col = self.limb_joint_indices.index(idx) if idx in self.limb_joint_indices else None
                    if col is not None and col < loss_nk.size(1):
                        loss_nk[:, col] = loss_nk[:, col] * self.mask_weight

        denom = (w_limb.sum() + EPS)
        base = loss_nk.sum() / denom

        a = self._alpha(epoch)
        return base * (self.loss_weight * a)