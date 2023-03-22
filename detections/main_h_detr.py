import os
import sys
import torch
import random
import pocket
import argparse
import torchvision
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

import warnings
warnings.simplefilter("ignore", UserWarning)

from tqdm import tqdm
from torchvision.ops.boxes import batched_nms
from torch.utils.data import (
    Dataset, DataLoader,
    DistributedSampler,
)

from vcoco.vcoco import VCOCO

from detr.util import box_ops
from detr.datasets import transforms as T

from h_detr.models import build_model

class Engine(pocket.core.DistributedLearningEngine):
    def __init__(self, net, criterion, train_loader, test_loader, postprocessor, max_norm, **kwargs):
        super().__init__(net, criterion, train_loader, **kwargs)
        self.max_norm = max_norm
        self.test_loader = test_loader
        self.postprocessor = postprocessor

    def _on_start(self):
        ap, rec = self.eval(self.postprocessor)
        if self._rank == 0:
            perf = [ap.mean().item(), rec.mean().item()]
            print(
                f"Epoch: {self._state.epoch} =>\t"
                f"mAP: {perf[0]:.4f}, mRec: {perf[1]:.4f}"
            )
            self.best_perf = perf[0]

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

    def _on_end_epoch(self):
        ap, rec = self.eval(self.postprocessor)
        if self._rank == 0:
            perf = [ap.mean().item(), rec.mean().item()]
            print(
                f"Epoch: {self._state.epoch} =>\t"
                f"mAP: {perf[0]:.4f}, mRec: {perf[1]:.4f}"
            )
            # Save checkpoints
            checkpoint = {
                'iteration': self._state.iteration,
                'epoch': self._state.epoch,
                'performance': perf,
                'model_state_dict': self._state.net.module.state_dict(),
                'optim_state_dict': self._state.optimizer.state_dict(),
            }
            if self._state.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(self._cache_dir, "latest.pth"))
            if perf[0] > self.best_perf:
                self.best_perf = perf[0]
                torch.save(checkpoint, os.path.join(self._cache_dir, "best.pth"))
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    @torch.no_grad()
    def eval(self, postprocessors, thresh=0.1):
        dataloader = self.test_loader
        net = self._state.net
        net.eval()

        associate = pocket.utils.BoxAssociation(min_iou=0.5)
        if self._rank == 0:
            meter = pocket.utils.DetectionAPMeter(
                80, algorithm='INT', nproc=10
            )
        num_gt = torch.zeros(80)
        for images, targets in tqdm(dataloader, disable=(self._world_size != 1)):
            images = pocket.ops.relocate_to_cuda(images)
            outputs = pocket.ops.relocate_to_cpu(net(images))

            scores_clt = []; preds_clt = []; labels_clt = []
            detections = postprocessors(
                outputs, torch.stack([t["size"] for t in targets])
            )
            for det, target in zip(detections, targets):
                scores, labels, boxes = det.values()
                keep = torch.nonzero(scores >= thresh).squeeze(1)
                scores = scores[keep]
                labels = labels[keep]
                boxes = boxes[keep]

                gt_boxes = target['boxes']
                # Denormalise ground truth boxes
                gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                h, w = target['size']
                scale_fct = torch.stack([w, h, w, h])
                gt_boxes *= scale_fct
                gt_labels = target['labels']

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

                scores_clt.append(scores)
                preds_clt.append(labels)
                labels_clt.append(binary_labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = torch.cat(pocket.utils.all_gather(scores_clt))
            preds_ddp = torch.cat(pocket.utils.all_gather(preds_clt))
            labels_ddp = torch.cat(pocket.utils.all_gather(labels_clt))

            if self._rank == 0:
                meter.append(scores_ddp, preds_ddp, labels_ddp)

        if self._world_size > 1:
            num_gt = num_gt.cuda()
            dist.barrier()
            dist.all_reduce(num_gt, op=dist.ReduceOp.SUM)
        if self._rank == 0:
            meter.num_gt = num_gt.tolist()
            ap = meter.eval()
            max_rec = meter.max_rec
            return ap, max_rec
        else:
            return -1, -1

class HICODetObject(Dataset):
    def __init__(self, dataset, transforms, nms_thresh=0.7):
        self.dataset = dataset
        self.transforms = transforms
        self.nms_thresh = nms_thresh
        self.conversion = [
             4, 47, 24, 46, 34, 35, 21, 59, 13,  1, 14,  8, 73, 39, 45, 50,  5,
            55,  2, 51, 15, 67, 56, 74, 57, 19, 41, 60, 16, 54, 20, 10, 42, 29,
            23, 78, 26, 17, 52, 66, 33, 43, 63, 68,  3, 64, 49, 69, 12,  0, 53,
            58, 72, 65, 48, 76, 18, 71, 36, 30, 31, 44, 32, 11, 28, 37, 77, 38,
            27, 70, 61, 79,  9,  6,  7, 62, 25, 75, 40, 22
        ]
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
        converted_labels = torch.as_tensor([self.conversion[i.item()] for i in labels])
        # Apply transform
        image, target = self.transforms(image, dict(boxes=boxes, labels=converted_labels))
        return image, target

class VCOCOObject(Dataset):
    def __init__(self, dataset, transformers, nms_thresh=0.7):
        self.dataset = dataset
        self.transforms = transformers
        self.nms_thresh = nms_thresh
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        labels = torch.cat([
            torch.ones_like(target['objects']),
            target['objects']
        ])
        # Convert to zero-based index
        labels -= 1

        keep = batched_nms(
            boxes, torch.ones(len(boxes)),
            labels, iou_threshold=self.nms_thresh
        )
        boxes = boxes[keep]
        labels = labels[keep]

        image, target = self.transforms(image, dict(boxes=boxes, labels=labels))
        return image, target

def initialise(args):
    # Load model and loss function
    detr, criterion, postprocessors = build_model(args)
    if os.path.exists(args.resume):
        print(f"Resume from model at {args.resume}")
        detr.load_state_dict(torch.load(args.resume)['model_state_dict'])
    elif os.path.exists(args.pretrained):
        print(f"Load pre-trained model from {args.pretrained}")
        model_weights = torch.load(args.pretrained)['model']
        keep = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90
        ]
        for k in model_weights.keys():
            if "class_embed" in k:
                model_weights[k] = model_weights[k][keep]
        detr.load_state_dict(model_weights)

    # Prepare dataset transforms
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    transforms_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ColorJitter(.4, .4, .4),
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
    transforms_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    # Load dataset
    # train_set = HICODetObject(
    #     pocket.data.HICODet(
    #         root=os.path.join(args.data_root, f'hico_20160224_det/images/train2015'),
    #         anno_file=os.path.join(args.data_root, f'instances_train2015.json'),
    #         target_transform=pocket.ops.ToTensor(input_format='dict')
    #     ), transforms_train
    # )
    # test_set = HICODetObject(
    #     pocket.data.HICODet(
    #         root=os.path.join(args.data_root, f'hico_20160224_det/images/test2015'),
    #         anno_file=os.path.join(args.data_root, f'instances_test2015.json'),
    #         target_transform=pocket.ops.ToTensor(input_format='dict')
    #     ), transforms_test
    # )

    train_set = VCOCOObject(VCOCO(
        root=os.path.join(args.data_root, "mscoco2014/train2014"),
        anno_file=os.path.join(args.data_root, "instances_vcoco_trainval.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ), transforms_train)
    test_set = VCOCOObject(VCOCO(
        root=os.path.join(args.data_root, "mscoco2014/val2014"),
        anno_file=os.path.join(args.data_root, "instances_vcoco_test.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    ), transforms_test)

    return detr, criterion, postprocessors['bbox'], [train_set, test_set]

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

    model, criterion, postprocessors, datasets = initialise(args)
    train_loader = DataLoader(
        dataset=datasets[0], collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            datasets[0], num_replicas=args.world_size,
            rank=rank, drop_last=True
        )
    )
    test_loader = DataLoader(
        dataset=datasets[1], collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            datasets[1], num_replicas=args.world_size,
            rank=rank, drop_last=True, shuffle=False
        )
    )

    engine = Engine(
        model, criterion, train_loader,
        test_loader, postprocessors,
        max_norm=args.clip_max_norm,
        print_interval=args.print_interval,
        cache_dir=args.output_dir
    )

    if args.eval:
        ap, rec = engine.eval(postprocessors)
        if rank == 0:
            print(f"The mAP is {ap.mean().item():.4f}, the mRec is {rec.mean().item():.4f}")
        return

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad],
            "lr": args.lr,
        }, {
            "params": [p for n, p in model.named_parameters()
            if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        }, {
            "params": [p for n, p in model.named_parameters()
            if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    engine.update_state_key(optimizer=optimizer, lr_scheduler=lr_scheduler)
    engine(args.epochs)

@torch.no_grad()
def sanity_check(args):
    model, criterion, postprocessors, datasets = initialise(args)
    image, target = datasets[0][0]
    model = model.cuda()
    image = image.cuda()
    print("\nPrinting out the detection target =>")
    for k, v in target.items():
        print(f"{k}: {v}")
    output = pocket.ops.relocate_to_cpu(model([image]))
    loss_dict = criterion(output, [target])
    print("\nPrinting out the computed losses =>")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")

    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    print("\nPrinting out the total loss =>")
    print(losses.item())

    scores, labels, boxes = postprocessors(output, target['size'].unsqueeze(0))[0].values()
    keep = torch.nonzero(scores >= 0.5).squeeze()
    if len(keep) == 0:
        print("No detections above score threshold.")
        sys.exit()
    print("\nPrinting out the detected instances =>")
    for c, s in zip(labels[keep], scores[keep]):
        print(f"Class {c.item()}: {s.item():.4f}")

    image = torchvision.transforms.ToPILImage()(image)
    image_copy = image.copy()
    pocket.utils.draw_boxes(image, boxes[keep], width=3)
    image.save("image.png")

    _, _, boxes = postprocessors(
        dict(
            pred_logits=torch.rand(1, len(target['boxes']), 80),
            pred_boxes=target['boxes'].unsqueeze(0)
        ), target['size'].unsqueeze(0)
    )[0].values()
    pocket.utils.draw_boxes(image_copy, boxes, width=3)
    image_copy.save("detections.png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

    # Arguments for fine-tuning
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--lr_backbone', default=2e-6, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')

    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Swin backbone
    parser.add_argument("--pretrained_backbone_path", default=None, type=str)
    parser.add_argument("--drop_path_rate", default=0.2, type=float)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--num_queries_one2one", default=300, type=int,
                        help="Number of query slots for one-to-one matching",)
    parser.add_argument("--num_queries_one2many", default=0, type=int,
                        help="Number of query slots for one-to-many matchining",)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Deformable DETR tricks
    parser.add_argument("--mixed_selection", action="store_true", default=False)
    parser.add_argument("--look_forward_twice", action="store_true", default=False)
    # Hybrid branch
    parser.add_argument("--k_one2many", default=6, type=int)
    parser.add_argument("--lambda_one2many", default=1.0, type=float)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # * Misc
    parser.add_argument('--world_size', default=8, type=int)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--sanity', default=False, action="store_true")
    parser.add_argument('--data_root', default="..", type=str)
    parser.add_argument('--pretrained', default='', help="load pretrained model", type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--print_interval', default=100, type=int)
    parser.add_argument('--output_dir', default='./checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--use_checkpoint", default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
