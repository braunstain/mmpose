from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DebugKeys(BaseTransform):
    def __init__(self, once=True):
        self.once = once
        self._printed = False
    def transform(self, results):
        if (not self.once) or (not self._printed):
            print("RESULTS KEYS:", sorted(results.keys()))
            if 'img' in results:
                import numpy as np
                img = results['img']
                print("IMG:", type(img), getattr(img, "shape", None), getattr(img, "dtype", None))
            self._printed = True
        return results