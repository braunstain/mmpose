from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class OcclusionWeighting(BaseTransform):

    def __init__(self, occ_weight=2.0):
        self.occ_weight = occ_weight

    def transform(self, results):

        kpt_vis = results['keypoints_visible']

        # COCO: 0=not labeled, 1=occluded, 2=visible
        occ_mask = (kpt_vis == 1)
        vis_mask = (kpt_vis == 2)

        weights = np.ones_like(kpt_vis, dtype=np.float32)

        weights[occ_mask] = self.occ_weight
        weights[kpt_vis == 0] = 0.0

        results['dataset_keypoint_weights'] = weights

        return results