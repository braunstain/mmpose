# Copyright (c) OpenMMLab. All rights reserved.
from html import parser
import logging
from argparse import ArgumentParser
from xml.parsers.expat import model

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import os
import json
import numpy as np

def xywh_to_xyxy(b):
    x, y, w, h = b
    print(x, y, x+w, y+h)
    return [x, y, x + w, y + h]

def load_gt_bboxes(ann_file, img_path, img_id=None):
    """Return Nx4 float32 array of GT person bboxes in xyxy format."""
    coco = json.load(open(ann_file, 'r', encoding='utf-8'))
    id = 0
    #print (coco.get('images', []))
    for img in coco.get('images', []):
        if img.get('file_name', '') == img_path or img.get('file_name', '') == img_id:
            id = img.get('id', None)
            print (img)
            break
    # Find the image id
    if img_id is None:
            raise ValueError(f'no img_id')


    # Collect bboxes for that image
    bboxes = []
    for ann in coco.get('annotations', []):
        #print (ann.get('file_name'), img_id)
        #print(ann)
        if ann.get('image_id') == id:
            print (ann)
            bbox = ann.get('bbox', None)
            bboxes.append(xywh_to_xyxy(bbox))
        

    if len(bboxes) == 0:
        return None, img_id

    return np.asarray(bboxes, dtype=np.float32), img_id

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
    '--ann-file',
    type=str,
    default=None,
    help='COCO-format annotation json file to load GT bboxes from')

    parser.add_argument(
    '--img-id',
    default=None,
    help='Image id in the COCO json (optional). If not set, match by file_name basename.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # inference a single image
    bboxes = None
    if args.ann_file is not None:
        bboxes, resolved_img_id = load_gt_bboxes(args.ann_file, args.img, args.img_id)
        print_log(f'Loaded {0 if bboxes is None else len(bboxes)} GT bboxes for image_id={resolved_img_id}', logger='current')

    # If bboxes is None, it falls back to the old single-person behavior
    batch_results = inference_topdown(model, args.img, bboxes)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)

    if args.out_file is not None:
        print_log(
            f'the output image has been saved at {args.out_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
