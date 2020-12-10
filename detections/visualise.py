"""
Visualize the detection results

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

import torch
from torchvision.ops import nms
from pocket.data import HICODet

def visualize(args):
    # Set up root directory
    partition = args.partition
    dataset = HICODet(None, os.path.join(
        args.data_root, 'instances_{}.json'.format(args.partition)
    ))
    # Set up image instance path
    image_name = dataset.filename(args.image_idx)
    print("Image name: ", image_name)
    image_path = os.path.join(
        args.data_root,
        'hico_20160224_det/images/{}'.format(args.partition),
        image_name
    )
    detection_path = os.path.join(
        args.detection_root,
        image_name.replace('.jpg', '.json')
    )
    # Load image instance
    image = Image.open(image_path)
    with open(detection_path, 'r') as f:
        detections = json.load(f)
    # Remove low-scoring boxes
    box_score_thresh = args.box_score_thresh
    boxes = np.asarray(detections['boxes'])
    scores = np.asarray(detections['scores'])
    keep_idx = np.where(scores >= box_score_thresh)[0]
    boxes = boxes[keep_idx, :]
    scores = scores[keep_idx]
    # Perform NMS
    keep_idx = nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        args.nms_thresh
    )
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    # Draw boxes
    canvas = ImageDraw.Draw(image)
    for idx in range(boxes.shape[0]):
        coords = boxes[idx, :].tolist()
        canvas.rectangle(coords)
        canvas.text(coords[:2], str(scores[idx])[:4])

    image.show()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize object detections")
    parser.add_argument('--detection-root', type=str, required=True)
    parser.add_argument('--image-idx', type=int, default=0)
    parser.add_argument('--data-root', type=str, default='../')
    parser.add_argument('--partition', type=str, default='train2015')
    parser.add_argument('--box-score-thresh', type=float, default=0.2)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    args = parser.parse_args()

    visualize(args)
