from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
import numpy as np
import random

@TRANSFORMS.register_module()
class LimbJointAugmentation(BaseTransform):
    """
    Paper-style Limb Joint Augmentation (LJA):
    - Focus only on VISIBLE limb joints (v==2)
    - Randomly select M = ceil(r * V) joints to occlude
    - Occlusion block size: (s * bbox_h, s * bbox_w)
    - Fill the whole block with a constant random value in [0,255]

    Apply BEFORE TopdownAffine.
    """

    # COCO-17 indices:
    # 0 nose, 1 leye, 2 reye, 3 lear, 4 rear,
    # 5 lsho, 6 rsho, 7 lel, 8 rel, 9 lwri, 10 rwri,
    # 11 lhip, 12 rhip, 13 lknee, 14 rknee, 15 lank, 16 rank
    DEFAULT_LIMB_IDS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    def __init__(self,
                 p=0.5,
                 occ_ratio=0.15,     # r in the paper
                 size_ratio=0.10,    # s in the paper
                 limb_ids=None,
                 only_visible=True,  # occlude only v==2 joints (paper does this)
                 rng_seed=21):
        self.p = float(p)
        self.occ_ratio = float(occ_ratio)
        self.size_ratio = float(size_ratio)
        self.limb_ids = tuple(limb_ids) if limb_ids is not None else self.DEFAULT_LIMB_IDS
        self.only_visible = bool(only_visible)
        self.rng = random.Random(rng_seed)

    @staticmethod
    def _as_kpts_vis(results):
        kpts = np.array(results.get('keypoints', None))
        vis = np.array(results.get('keypoints_visible', None))

        if kpts.size == 0 or vis.size == 0:
            return None, None

        # Ensure shapes: kpts -> (K,2), vis -> (K,)
        if kpts.ndim == 3:
            kpts = kpts[0]
        if vis.ndim == 2:
            vis = vis[0]

        return kpts, vis

    @staticmethod
    def _bbox_hw(results):
        bbox = results.get('bbox', None)
        if bbox is None:
            return None, None

        bbox = np.array(bbox, dtype=np.float32)

        # Common cases:
        # (4,) -> [x,y,w,h]
        # (1,4) or (N,4) -> take first row
        if bbox.ndim == 2:
            if bbox.shape[0] < 1 or bbox.shape[1] < 4:
                return None, None
            bbox = bbox[0]
        elif bbox.ndim != 1:
            return None, None

        if bbox.size < 4:
            return None, None

        w = float(bbox[2])
        h = float(bbox[3])
        return h, w

    def transform(self, results):
        if self.rng.random() > self.p:
            return results
        if 'img' not in results:
            return results

        img = results['img']
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            return results

        H, W = img.shape[:2]
        kpts, vis = self._as_kpts_vis(results)
        if not hasattr(self, "_dbg"):
            self._dbg = True
            print("vis unique:", np.unique(vis), "vis shape:", vis.shape)
        if kpts is None:
            return results

        bbox_h, bbox_w = self._bbox_hw(results)
        if bbox_h is None:
            # Fallback: use image dims if bbox missing (shouldn’t happen in your pipeline)
            bbox_h, bbox_w = float(H), float(W)

        # Candidate limb joints
        limb_ids = np.array(self.limb_ids, dtype=np.int64)
        limb_ids = limb_ids[(limb_ids >= 0) & (limb_ids < len(vis))]

        cand = limb_ids[vis[limb_ids] > 0]

        V = int(len(cand))
        if V == 0:
            return results

        M = int(np.ceil(self.occ_ratio * V))
        M = max(1, min(M, V))

        chosen = self.rng.sample(list(cand), k=M)

        occ_h = max(2, int(self.size_ratio * bbox_h))
        occ_w = max(2, int(self.size_ratio * bbox_w))

        for j in chosen:
            x, y = kpts[j]
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            cx = int(np.clip(round(x), 0, W - 1))
            cy = int(np.clip(round(y), 0, H - 1))

            x1 = int(np.clip(cx - occ_w // 2, 0, W - 1))
            y1 = int(np.clip(cy - occ_h // 2, 0, H - 1))
            x2 = int(np.clip(x1 + occ_w, 0, W))
            y2 = int(np.clip(y1 + occ_h, 0, H))

            if x2 <= x1 or y2 <= y1:
                continue

            # constant random value in [0,255] (paper)
            v = self.rng.randint(0, 255)
            img[y1:y2, x1:x2] = v

        results['img'] = img
        return results