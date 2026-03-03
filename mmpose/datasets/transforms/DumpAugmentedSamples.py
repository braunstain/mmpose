from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
import os
import cv2
import numpy as np

@TRANSFORMS.register_module()
class DumpAugmentedSamples(BaseTransform):
    def __init__(self, out_dir='work_dirs/aug_debug', max_save=20):
        self.out_dir = out_dir
        self.max_save = max_save
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)

    def transform(self, results):
        if self.count >= self.max_save:
            return results
        if 'img' not in results:
            return results

        img = results['img']
        if not isinstance(img, np.ndarray):
            return results

        # draw keypoints (optional)
        kpts = results.get('keypoints', None)
        vis = results.get('keypoints_visible', None)
        if kpts is not None and vis is not None:
            k = np.array(kpts)
            v = np.array(vis)
            if k.ndim == 3:
                k = k[0]
            if v.ndim == 2:
                v = v[0]
            for (x, y), vv in zip(k, v):
                if vv == 0:
                    continue
                # green for visible, red for occluded
                color = (0, 255, 0) if vv == 2 else (0, 0, 255)
                cv2.circle(img, (int(x), int(y)), 3, color, -1)

        fname = f"{self.count:03d}_pre_affine.jpg"
        cv2.imwrite(os.path.join(self.out_dir, fname), img)
        self.count += 1
        return results