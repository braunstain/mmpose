import json, math

ann_path = r"data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"
with open(ann_path, "r") as f:
    coco = json.load(f)

areas = []
for a in coco["annotations"]:
    # COCO keypoint area bucket uses bbox area; your file already has "area"
    areas.append(a.get("area", a["bbox"][2] * a["bbox"][3]))

medium = sum(1024 <= x < 9216 for x in areas)
large  = sum(x >= 9216 for x in areas)

print("total anns:", len(areas))
print("medium:", medium)
print("large:", large)
print("min area:", min(areas), "max area:", max(areas))