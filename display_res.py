import pickle
import numpy as np

with open("OCHuman_val_dump.pkl", "rb") as f:
    data = pickle.load(f)

pi = data[0]['pred_instances']
print("type(pred_instances):", type(pi))
print("dir(pred_instances) sample:", [a for a in dir(pi) if not a.startswith('_')][:40])

# Try common access patterns:
for attr in ['keypoints','keypoint_scores','scores','bboxes','bbox_scores']:
    v = getattr(pi, attr, None)
    if v is not None:
        try:
            import numpy as np
            arr = v.detach().cpu().numpy() if hasattr(v,'detach') else (v.cpu().numpy() if hasattr(v,'cpu') else v)
            print(attr, "shape:", getattr(arr,'shape',None), "dtype:", getattr(arr,'dtype',None))
        except Exception as e:
            print(attr, "exists but couldn't convert:", e)

# Some MMEngine InstanceData stores as dict-like too:
try:
    print("pred_instances keys:", pi.keys())
except Exception as e:
    print("pred_instances not dict-like:", e)
"""
vis_err = []
occ_err = []

for s in data:
    gt = np.array(s["raw_ann_info"]["keypoints"]).reshape(17,3)
    gt_xy = gt[:, :2]
    v = gt[:, 2]

    pred_xy = np.array(s["pred_instances"]["keypoints"])[0]  # (17,2)

    # normalize by bbox size (so errors comparable across scales)
    x,y,w,h = s["raw_ann_info"]["bbox"]
    norm = max(w, h)

    dist = np.linalg.norm(pred_xy - gt_xy, axis=1) / (norm + 1e-9)

    vis_err.extend(dist[v == 2])
    occ_err.extend(dist[v == 1])

vis_err = np.array(vis_err, dtype=float)
occ_err = np.array(occ_err, dtype=float)

print("Visible mean norm error:", vis_err.mean(), "count", len(vis_err))
print("Occluded mean norm error:", occ_err.mean(), "count", len(occ_err))
print("Error ratio occ/vis:", occ_err.mean()/vis_err.mean())

"""