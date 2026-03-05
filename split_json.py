import json, math


def build_subset(anns, data, out_file):

    img_ids = {ann["image_id"] for ann in anns}

    images = [img for img in data["images"] if img["id"] in img_ids]

    subset = {
        "images": images,
        "annotations": anns,
        "categories": data["categories"]
    }
    print(subset.keys())
    with open(out_file, "w") as f:
        json.dump(subset, f)

ann_path = r"data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"
with open(ann_path, "r") as f:
    coco = json.load(f)

extreme_anns_occ_count = 0
high_occ_count = 0
med_occ_count = 0
low_occ_count = 0
low_anns = []
mid_anns = []
high_anns = []
extreme_anns = []

len = len(coco["annotations"])
for ann in coco["annotations"]:
    keypoints = ann["keypoints"]
    vis = keypoints[2::3]

    sum_miss = sum(v == 0 for v in vis)
    sum_occ = sum(v == 1 for v in vis)
    sum_vis = sum(v == 2 for v in vis)

    if (sum_vis + sum_occ) > 0:
        if sum_occ / (sum_vis + sum_occ) >= 0.5:
            extreme_anns_occ_count += 1
            extreme_anns.append(ann)
        if sum_occ / (sum_vis + sum_occ) >= 0.3:
            high_occ_count += 1
            high_anns.append(ann)
        elif sum_occ / (sum_vis + sum_occ) >= 0.1:
            med_occ_count += 1
            mid_anns.append(ann)    
        else:
            low_occ_count += 1
            low_anns.append(ann)

print(f"Total annotations: {len}")
print(f"Extreme occlusion: {extreme_anns_occ_count}")
print(f"High occlusion: {high_occ_count}")
print(f"Medium occlusion: {med_occ_count}")
print(f"Low occlusion: {low_occ_count}")

build_subset(low_anns, coco, "data/OCHuman/annotations/ochuman_low.json")
build_subset(mid_anns, coco, "data/OCHuman/annotations/ochuman_mid.json")
build_subset(high_anns, coco, "data/OCHuman/annotations/ochuman_high.json")
build_subset(extreme_anns, coco, "data/OCHuman/annotations/ochuman_extreme.json")

