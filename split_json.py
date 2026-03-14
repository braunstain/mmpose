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

ann_path = r"data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json"
with open(ann_path, "r") as f:
    coco = json.load(f)

extreme_anns_occ_count = 0
high_occ_count = 0
med_occ_count = 0
low_occ_count = 0
zero_occ_count = 0
non_zero_occ_count = 0
one_count = 0
more_one_count = 0
non_zero_anns = []
zero_anns = []
low_anns = []
mid_anns = []
high_anns = []
extreme_anns = []
one_anns = []
more_anns = []

len = len(coco["annotations"])
for ann in coco["annotations"]:
    keypoints = ann["keypoints"]
    vis = keypoints[2::3]

    sum_miss = sum(v == 0 for v in vis)
    sum_occ = sum(v == 1 for v in vis)
    sum_vis = sum(v == 2 for v in vis)

    if (sum_vis + sum_occ) > 0:
        if sum_occ == 0:
            zero_occ_count += 1
            zero_anns.append(ann)
        if sum_occ > 0:
            non_zero_occ_count += 1
            non_zero_anns.append(ann)
        if sum_occ == 1:
            one_count += 1
            one_anns.append(ann)
        if sum_occ > 1:
            more_one_count += 1
            more_anns.append(ann)

print(f"Total annotations: {len}")
print(f"One occlusion: {one_count}")
print(f"More than one occlusion: {more_one_count}")
print(f"Zero occlusion: {zero_occ_count}")
print(f"Non-zero occlusion: {non_zero_occ_count}")

build_subset(non_zero_anns, coco, "data/OCHuman/annotations/test_split/ochuman_test_non_zero.json")
build_subset(zero_anns, coco, "data/OCHuman/annotations/test_split/ochuman_test_zero.json")
build_subset(one_anns, coco, "data/OCHuman/annotations/test_split/ochuman_test_one.json") 
build_subset(more_anns, coco, "data/OCHuman/annotations/test_split/ochuman_test_more_than_one.json")

