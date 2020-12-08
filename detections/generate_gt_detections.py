"""
Generate ground truth detection boxes
for each image and save as .json files 

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import argparse
from tqdm import tqdm

from pocket.data import HICODet

def main(args, human_idx=49):
    dataset = HICODet(None, os.path.join(
        args.data_root,
        'instances_{}.json'.format(args.partition)
    ))
    cache_dir = os.path.join(args.cache_dir, '{}_gt'.format(args.partition))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for i, anno in enumerate(tqdm(dataset._anno)):
        fname = dataset._filenames[i]
        npairs = len(anno['boxes_h'])

        boxes = anno['boxes_h'] + anno['boxes_o']
        labels = [human_idx for _ in range(npairs)] + anno['object']
        scores = [1. for _ in range(2 * npairs)]

        with open(os.path.join(
            cache_dir,
            fname.replace('.jpg', '.json')),
        'w') as f:
            json.dump(dict(boxes=boxes, labels=labels, scores=scores), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate ground truth detections")
    parser.add_argument('--data-root', type=str, default='../')
    parser.add_argument('--partition', type=str, default='train2015')
    parser.add_argument('--cache-dir', type=str, default='./')

    args = parser.parse_args()
    
    print(args)
    main(args)