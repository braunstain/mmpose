import json, math
import numpy as np
from collections import defaultdict

# ---- load data ----
gt_path = r"data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json"
pred_path = r"pred_coco_results.json"  # list[ {image_id, keypoints, score, ...} ]

gt = json.load(open(gt_path, "r", encoding="utf-8"))
preds = json.load(open(pred_path, "r", encoding="utf-8"))

# map image_id -> file_name
imgid2file = {im["id"]: im.get("file_name", "") for im in gt["images"]}

# map image_id -> GT anns (people)
gts_by_img = defaultdict(list)
for ann in gt["annotations"]:
    gts_by_img[ann["image_id"]].append(ann)

# map image_id -> preds (people)
preds_by_img = defaultdict(list)
for p in preds:
    preds_by_img[p["image_id"]].append(p)

# ---- OKS helper (occluded-only), used for matching ----
# If you already have vars_ from COCO sigmas, plug it here.
# Otherwise, use standard COCO sigmas for your K (17 for COCO).
COCO_SIGMAS_17 = np.array(
    [0.26,0.25,0.25,0.35,0.35,0.79,0.79,0.72,0.72,0.62,0.62,1.07,1.07,0.87,0.87,0.89,0.89],
    dtype=np.float32
) / 10.0
vars_ = (COCO_SIGMAS_17 * 2.0) ** 2

def oks_occluded(dt_kpts, gt_kpts, area):
    g = np.asarray(gt_kpts, dtype=np.float32)
    gx, gy, gv = g[0::3], g[1::3], g[2::3]
    m = (gv == 1)  # occluded only
    if not np.any(m):
        return None  # OKS undefined if no occluded joints

    d = np.asarray(dt_kpts, dtype=np.float32)
    dx, dy = d[0::3], d[1::3]

    dd = (dx[m] - gx[m])**2 + (dy[m] - gy[m])**2
    e = dd / (vars_[m] * (area + np.spacing(1)) * 2.0)
    return float(np.mean(np.exp(-e)))

def norm_occ_error(dt_kpts, gt_kpts, area):
    """Mean normalized L2 error over occluded joints: sqrt(dd/(area)) averaged."""
    g = np.asarray(gt_kpts, dtype=np.float32)
    gx, gy, gv = g[0::3], g[1::3], g[2::3]
    m = (gv == 1)
    if not np.any(m):
        return None

    d = np.asarray(dt_kpts, dtype=np.float32)
    dx, dy = d[0::3], d[1::3]

    dd = (dx[m] - gx[m])**2 + (dy[m] - gy[m])**2
    # normalize by person scale; sqrt to be in "pixels normalized by sqrt(area)"
    return float(np.mean(np.sqrt(dd / (area + 1e-9))))

def ann_area(ann):
    a = ann.get("area", None)
    if a is not None and a > 0:
        return float(a)
    bbox = ann.get("bbox", [0,0,1,1])
    a = float(bbox[2] * bbox[3])
    return a if a > 0 else 1.0

# ---- greedy matching per image (by pred score) ----
def match_image(pred_list, gt_list, oks_thr=0.1):
    """
    Returns list of matched pairs (pred, gt, oks)
    Matching: sort preds by score desc, assign each to best unmatched gt by occluded-OKS.
    """
    pred_list = sorted(pred_list, key=lambda x: x.get("score", 0.0), reverse=True)
    used = set()
    matches = []

    # precompute OKS matrix (only for GTs where occluded exists)
    for p in pred_list:
        best_j, best_oks = None, -1.0
        for j, gt_ann in enumerate(gt_list):
            if j in used:
                continue
            o = oks_occluded(p["keypoints"], gt_ann["keypoints"], ann_area(gt_ann))
            if o is None:
                continue
            if o > best_oks:
                best_oks, best_j = o, j
        if best_j is not None and best_oks >= oks_thr:
            used.add(best_j)
            matches.append((p, gt_list[best_j], best_oks))
    return matches

# ---- score each image by "worst occluded error among matched persons" ----
image_scores = []
for image_id, gt_list in gts_by_img.items():
    pred_list = preds_by_img.get(image_id, [])
    if not pred_list:
        continue

    matches = match_image(pred_list, gt_list, oks_thr=0.1)

    # compute occluded error per match; take max (worst person) as image severity
    errs = []
    for p, gt_ann, _oks in matches:
        e = norm_occ_error(p["keypoints"], gt_ann["keypoints"], ann_area(gt_ann))
        if e is not None:
            errs.append(e)

    if not errs:
        continue

    score = max(errs)  # worst person in the image
    image_scores.append((score, image_id, imgid2file.get(image_id, "")))

# rank worst
image_scores.sort(reverse=True, key=lambda x: x[0])
top10 = image_scores[:10]

for rank, (score, image_id, fname) in enumerate(top10, 1):
    print(f"{rank:02d}) image_id={image_id}  score={score:.4f}  file={fname}")