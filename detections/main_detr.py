import enum
import os
import json
import torch
import random
import pocket
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from torchvision.ops.boxes import batched_nms
from torch.utils.data import (
    Dataset, DataLoader,
    DistributedSampler, BatchSampler
)

import sys
sys.path.append('detr')

from util import box_ops
from models import build_model
import datasets.transforms as T

class Engine(pocket.core.DistributedLearningEngine):
    def __init__(self, net, criterion, dataloader, max_norm, **kwargs):
        super().__init__(net, criterion, dataloader, **kwargs)
        self.max_norm = max_norm

    def _on_start_epoch(self):
        self._state.epoch += 1
        self._state.net.train()
        self._train_loader.batch_sampler.sampler.set_epoch(self._state.epoch)

    def _on_each_iteration(self):
        self._state.output = self._state.net(*self._state.inputs)
        loss_dict = self._criterion(self._state.output, self._state.targets)
        weight_dict = self._criterion.weight_dict
        self._state.loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def eval(self, postprocessors, thresh=0.1):
        self._state.net.eval()
        associate = pocket.utils.BoxAssociation(min_iou=0.5)
        meter = pocket.utils.DetectionAPMeter(
            80, algorithm='INT', nproc=10
        )
        num_gt = torch.zeros(80)
        if self._train_loader.batch_size != 1:
            raise ValueError(f"The batch size shoud be 1, not {self._train_loader.batch_size}")
        for image, target in tqdm(self._train_loader):
            image = pocket.ops.relocate_to_cuda(image)
            output = self._state.net(image)
            output = pocket.ops.relocate_to_cpu(output)
            scores, labels, boxes = postprocessors(
                output, target[0]['size'].unsqueeze(0)
            )[0].values()
            keep = torch.nonzero(scores >= thresh).squeeze(1)
            scores = scores[keep]
            # Convert to one-based index
            labels = labels[keep] + 1
            boxes = boxes[keep]

            gt_boxes = target[0]['boxes']
            gt_labels = target[0]['labels']

            for c in gt_labels:
                num_gt[c] += 1

            # Associate detections with ground truth
            binary_labels = torch.zeros(len(labels))
            unique_cls = labels.unique()
            for c in unique_cls:
                det_idx = torch.nonzero(labels == c).squeeze(1)
                gt_idx = torch.nonzero(gt_labels == c).squeeze(1)
                if len(gt_idx) == 0:
                    continue
                binary_labels[det_idx] = associate(
                    gt_boxes[gt_idx].view(-1, 4),
                    boxes[det_idx].view(-1, 4),
                    scores[det_idx].view(-1)
                )

            meter.append(scores, labels, binary_labels)

        meter.num_gt = num_gt.tolist()
        return meter.eval(), meter.max_rec

class HICODetObject(Dataset):
    def __init__(self, dataset, data_root, transforms, nms_thresh=0.7):
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
        # Remove overlapping ground truth boxes
        keep = batched_nms(
            boxes, torch.ones(len(boxes)),
            labels, iou_threshold=self.nms_thresh
        )
        boxes = boxes[keep]
        labels = labels[keep]
        # Convert HICODet object indices to COCO indices
        converted_labels = torch.tensor([int(self.hico2coco[i.item()]) for i in labels])
        # Apply transform
        image, target = self.transforms(image, dict(boxes=boxes, labels=converted_labels))
        return image, target

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
        print(f"Load pre-trained model from {args.pretrained}")
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

def collate_fn(batch):
    images = []; targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    model, criterion, postprocessors, dataset = initialise(args)
    if args.eval:
        sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler,
            batch_size=1, collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=False
        )
    else:
        sampler = DistributedSampler(dataset)
        batch_sampler = BatchSampler(
            sampler, args.batch_size,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=args.num_workers
        )

    engine = Engine(
        model, criterion, dataloader,
        max_norm=args.clip_max_norm,
        print_interval=args.print_interval,
        cache_dir=args.output_dir
    )

    if args.eval:
        ap, rec = engine.eval(postprocessors)
        print(f"The mAP is {ap.mean().item():.4f}, the mRec is {rec.mean().item():.4f}")
    else:
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad]
            }, {
                "params": [p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr,
            weight_decay=args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        engine.update_state_key(optimizer=optimizer, lr_scheduler=lr_scheduler)
        engine(args.epochs)

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
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--partition', default='train2015')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--data_root', default='../')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained', default='', help='Start from a pre-trained model')
    parser.add_argument('--eval', action='store_true')

    # distributed training parameters
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--print-interval', default=1000, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')

    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
    
    # detr.eval()
    # image = Image.open('/Users/fredzzhang/Desktop/cellphone.jpeg')
    # image = Image.open('/Users/fredzzhang/Developer/github/hicodet/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg')

    # img, tgt = dataset[0]
    # print(tgt)
    # out = detr(img)
    # loss_dict = criterion(out, tgt)
    # for k, v in loss_dict.items():
    #     print(k, v)

    # weight_dict = criterion.weight_dict
    # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    # print(losses)

    # scores, labels, boxes = postprocessors(out, tgt[0]['size'].unsqueeze(0))[0].values()
    # keep = torch.nonzero(torch.logical_and(scores >= 0.9, labels != 0)).squeeze()
    # print(scores[keep])
    # print(labels[keep])

    # image = torchvision.transforms.ToPILImage()(img[0])
    # _, _, boxes = postprocessors(dict(pred_logits=torch.rand(1, 3, 81), pred_boxes=tgt[0]['boxes'].unsqueeze(0)), tgt[0]['size'].unsqueeze(0))[0].values()
    # pocket.utils.draw_boxes(image, boxes)
    # image.show()
