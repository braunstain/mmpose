import os
import json
import pickle

ann_file = 'data/train2017/annotations/instances_val_person2017.json'
img_root = 'data/train2017/val2017/'
out_file = 'data/train2017/coco_donor_pool.pkl'

with open(ann_file, 'r') as f:
    coco = json.load(f)

id_to_name = {img['id']: img['file_name'] for img in coco['images']}

donors = []
for ann in coco['annotations']:
    if ann.get('iscrowd', 0) != 0:
        continue
    if 'segmentation' not in ann or not ann['segmentation']:
        continue

    x, y, w, h = ann['bbox']
    area = ann.get('area', w * h)

    # Reject tiny and absurdly large donors
    if w < 25 or h < 40:
        continue
    if area < 1500:
        continue

    img_name = id_to_name[ann['image_id']]
    donors.append({
        'img_path': os.path.join(img_root, img_name),
        'bbox': ann['bbox'],
        'segmentation': ann['segmentation'],
        'area': area,
    })

with open(out_file, 'wb') as f:
    pickle.dump(donors, f)

print(f'Saved {len(donors)} donors to {out_file}')