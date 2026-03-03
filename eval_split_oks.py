import json
import os
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---- Part A: COCOeval subclass that masks OKS by visibility group ----
class COCOevalSplitVis(COCOeval):
    """
    COCOeval but with OKS computed on a subset of GT keypoints:
      mode="visible"  -> only v == 2
      mode="occluded" -> only v == 1
    """
    def __init__(self, cocoGt=None, cocoDt=None, iouType="keypoints", mode="visible"):
        super().__init__(cocoGt, cocoDt, iouType)
        assert mode in ("visible", "occluded")
        self.mode = mode

    def computeOks(self, imgId, catId):
        p = self.params
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        if len(gts) == 0 or len(dts) == 0:
            return []

        # sort detections by score, keep maxDets
        dts = sorted(dts, key=lambda x: -x["score"])[: p.maxDets[-1]]

        sigmas = p.kpt_oks_sigmas
        vars_ = (sigmas * 2) ** 2

        ious = np.zeros((len(dts), len(gts)), dtype=np.float32)

        for j, gt in enumerate(gts):
            g = np.asarray(gt["keypoints"], dtype=np.float32)
            gx = g[0::3]
            gy = g[1::3]
            gv = g[2::3]

            if self.mode == "visible":
                m = (gv == 2)
            else:
                m = (gv == 1)

            # if no keypoints in this group, OKS undefined => leave zeros
            if not np.any(m):
                continue

            area = gt.get("area", None)
            if area is None:
                # bbox is [x,y,w,h]
                bbox = gt.get("bbox", [0, 0, 1, 1])
                area = float(bbox[2] * bbox[3])
            if area <= 0:
                area = 1.0

            for i, dt in enumerate(dts):
                d = np.asarray(dt["keypoints"], dtype=np.float32)
                dx = d[0::3]
                dy = d[1::3]

                dd = (dx[m] - gx[m]) ** 2 + (dy[m] - gy[m]) ** 2
                e = dd / (vars_[m] * (area + np.spacing(1)) * 2.0)
                
                ious[i, j] = float(np.sum(np.exp(-e)) / e.shape[0])

        return ious




def filter_gt_by_mode(gt_json: str, mode: str, out_path: str):
    import json
    import numpy as np

    assert mode in ("visible", "occluded")
    data = json.load(open(gt_json, "r", encoding="utf-8"))

    keep_anns = []
    keep_img_ids = set()

    for ann in data["annotations"]:
        v = np.array(ann["keypoints"])[2::3]
        if mode == "visible":
            ok = np.any(v == 2)
        else:
            ok = np.any(v == 1)
        if ok:
            keep_anns.append(ann)


    data["annotations"] = keep_anns

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def count_people():
    gt_path = r"data\OCHuman\annotations\ochuman_coco_format_test_range_0.00_1.00.json"
    gt = json.load(open(gt_path,'r',encoding='utf-8'))

    no_vis = 0
    no_occ = 0
    total = 0
    visible_keypoints = 0
    occluded_keypoints = 0
    for ann in gt['annotations']:
        k = np.array(ann['keypoints'])[2::3]  # v flags
        total += 1
        if not np.any(k==2): no_vis += 1
        if not np.any(k==1): no_occ += 1
        for v in k:
            if v == 2:
                visible_keypoints += 1
            elif v == 1:
                occluded_keypoints += 1

    print("Total persons:", total)
    print("No visible (no v==2):", no_vis, f"({100*no_vis/total:.2f}%)")
    print("No occluded (no v==1):", no_occ, f"({100*no_occ/total:.2f}%)")   
    print("Visible keypoints:", visible_keypoints)
    print("Occluded keypoints:", occluded_keypoints)
# ---- Part B: convert a typical MMPose PKL dump -> COCO results list ----
def pkl_to_coco_results(pkl_path: str, category_id: int = 1):
    """
    Tries to convert common MMPose test/eval pkl formats into COCO 'keypoints' results.
    Output: list of dicts: {image_id, category_id, keypoints, score}
    """
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    results = []

    # Common cases:
    # 1) data is list, each element corresponds to one image
    # 2) each element has fields like:
    #    - 'img_id' or 'image_id'
    #    - 'pred_instances' with 'keypoints' and 'keypoint_scores' and maybe 'bbox_scores'
    #
    # We handle several variants defensively.

    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Unsupported PKL root type: {type(data)}")

    for item in data:
        if not isinstance(item, dict):
            continue

        image_id = item.get("img_id", item.get("image_id", None))
        if image_id is None:
            # some formats store it under metainfo
            meta = item.get("metainfo", item.get("meta", {}))
            image_id = meta.get("img_id", meta.get("image_id", None))
        if image_id is None:
            continue

        pred = item.get("pred_instances", item.get("preds", None))
        if pred is None:
            continue

        # Extract keypoints: expected shape [num_person, K, 2]
        kpts = pred.get("keypoints", None)
        if kpts is None:
            continue
        kpts = np.asarray(kpts)

        # Scores per keypoint: [num_person, K] (optional but common)
        kp_scores = pred.get("keypoint_scores", pred.get("keypoints_score", None))
        if kp_scores is not None:
            kp_scores = np.asarray(kp_scores)

        # Person score: sometimes bbox_score or instance_score exists
        inst_scores = pred.get("bbox_scores", pred.get("scores", pred.get("score", None)))
        if inst_scores is not None:
            inst_scores = np.asarray(inst_scores).reshape(-1)
        else:
            inst_scores = None

        num_person = kpts.shape[0]
        for n in range(num_person):
            xy = kpts[n]  # [K,2]
            K = xy.shape[0]

            # COCO result "keypoints" wants [x,y,score] per kpt
            flat = []
            for k in range(K):
                x, y = float(xy[k, 0]), float(xy[k, 1])
                s = float(kp_scores[n, k]) if kp_scores is not None else 1.0
                flat.extend([x, y, s])

            # detection score: prefer instance score if exists, else mean kp score
            if inst_scores is not None and n < len(inst_scores) and kp_scores is not None:
                score = float(inst_scores[n]) * float(np.mean(kp_scores[n]))
            elif kp_scores is not None:
                score = float(np.mean(kp_scores[n]))
            elif inst_scores is not None and n < len(inst_scores):
                score = float(inst_scores[n])
            else:
                score = 1.0

            results.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "keypoints": flat,
                "score": score,
            })

    if len(results) == 0:
        raise RuntimeError(
            "Conversion produced 0 detections. Your PKL structure likely differs. "
            "Print one item and we’ll adapt the parser."
        )

    return results


def eval_split_ap(gt_json: str, coco_results: list):
    cocoGt = COCO(gt_json)

    # loadRes expects a file OR list; list works.
    cocoDt = cocoGt.loadRes(coco_results)

    out = {}
    for mode in ("visible", "occluded"):
        tmp = f"tmp_gt_{mode}.json"
        filter_gt_by_mode(gt_json, mode, tmp)

        cocoGt = COCO(tmp)
        cocoDt = cocoGt.loadRes(coco_results)

        E = COCOevalSplitVis(cocoGt, cocoDt, iouType="keypoints", mode=mode)
        E.evaluate(); E.accumulate(); E.summarize()
        out[mode] = float(E.stats[0])

        os.remove(tmp)

    ap_vis = out["visible"]
    ap_occ = out["occluded"]
    print("\nSPLIT RESULTS")
    print(f"AP_visible  = {ap_vis:.4f}")
    print(f"AP_occluded = {ap_occ:.4f}")
    print(f"Gap         = {ap_vis - ap_occ:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="GT COCO keypoints json (OCHuman annotations)")
    parser.add_argument("--pkl", required=True, help="MMPose PKL dump")
    parser.add_argument("--out", default="pred_coco_results.json", help="Where to write converted results json")
    args = parser.parse_args()

    coco_results = pkl_to_coco_results(args.pkl, category_id=1)
    Path(args.out).write_text(json.dumps(coco_results))
    print(f"Wrote {len(coco_results)} detections to {args.out}")

    eval_split_ap(args.gt, coco_results)

    count_people()