import os
import json
import torch
import pocket
import argparse
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('detr')

from util import box_ops
from models import build_model
import datasets.transforms as T

class HICODetObject(Dataset):
    def __init__(self, dataset, data_root, transforms, nms_thresh=0.5):
        self.dataset = dataset
        self.transforms = transforms
        self.nms_thresh = nms_thresh
        with open(os.path.join(data_root, 'coco80tohico80.json'), 'r') as f:
            corr = json.load(f)
        self.hico2coco = dict(zip(corr.values(), corr.keys()))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        # Convert ground truth boxes to zero-based index and the
        # representation from pixel indices to coordinates
        boxes[:, :2] -= 1
        labels = torch.cat([
            49 * torch.ones_like(target['object']),
            target['object']
        ])
        # Convert HICODet object indices to COCO indices
        converted_labels = torch.tensor([int(self.hico2coco[i.item()]) for i in labels])
        # Apply transform
        image, target = self.transforms(image, dict(boxes=boxes, labels=converted_labels))
        return [image], [target]

class PostProcess(torch.nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

def initialise(args):
    # Load model and loss function
    detr, criterion, _ = build_model(args)
    if os.path.exists(args.pretrained):
        detr.load_state_dict(torch.load(args.pretrained)['model'])
    class_embed = torch.nn.Linear(256, 81, bias=True)
    w, b = detr.class_embed.state_dict().values()
    keep = [
        91, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        84, 85, 86, 87, 88, 89, 90
    ]
    # Remove deprecated classes
    class_embed.load_state_dict(dict(
        weight=w[keep], bias=b[keep]
    ))
    detr.class_embed = class_embed

    # Prepare dataset transforms
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if args.partition == 'train2015':
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if args.partition == 'test2015':
        transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    # Load dataset
    dataset = HICODetObject(
        pocket.data.HICODet(
            root=os.path.join(args.data_root, f'hico_20160224_det/images/{args.partition}'),
            anno_file=os.path.join(args.data_root, f'instances_{args.partition}.json'),
            target_transform=pocket.ops.ToTensor(input_format='dict')
        ), args.data_root, transforms
    )

    return detr, criterion, PostProcess(), dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
                            
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--partition', default='train2015')
    parser.add_argument('--data_root', default='../')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained', default='', help='Start from a pre-trained model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    detr, criterion, postprocessors = initialise(args)
    image = Image.open('/Users/fredzzhang/Desktop/cellphone.jpeg')
    # image = Image.open('/Users/fredzzhang/Developer/github/hicodet/hico_20160224_det/images/train2015/HICO_train2015_00000001.jpg')
    out = detr([pocket.ops.to_tensor(image, 'pil')])

    scores, labels, boxes = postprocessors(out, torch.as_tensor([image.height, image.width]).unsqueeze(0))[0].values()
    keep = torch.nonzero(torch.logical_and(scores >= 0.9, labels != 0)).squeeze()
    print(scores[keep])
    print(labels[keep])
    pocket.utils.draw_boxes(image, boxes[keep])
    image.show()