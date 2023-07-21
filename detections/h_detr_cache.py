import os
import json
import torch
import pocket
import argparse
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

from tqdm import tqdm
from detr.datasets import transforms as T

from h_detr.models import build_model

def initialise(args):
    # Load model and loss function
    detr, _, postprocessors = build_model(args)
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
    transforms = T.Compose([T.RandomResize([800], max_size=1333), normalize,])
    """
    Instructions on custom datasets

    1. The dataset class needs to have a __getitem__ method that returns an image (torch.Tensor)
    and a target dictionary with the key "size" that stores the shape of the image.
    2. The dataset class needs to have a filename() method that takes an index (integer) as input
    and outputs the name of the image file corresponding to the image. The cached detections will
    then be stored in a .json file with the same name.
    """
    dataset = pocket.data.HICODet(
        root=os.path.join(args.data_root, f'hico_20160224_det/images/test2015'),
        anno_file=os.path.join(args.data_root, f'instances_test2015.json'),
        transforms=transforms,
    )

    return detr, postprocessors['bbox'], dataset

def collate_fn(batch):
    images = []; targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets

@torch.no_grad()
def main(args):

    model, postprocessors, dataset = initialise(args)
    model.eval()
    model = model.to(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for idx, (image, target) in enumerate(tqdm(dataset)):
        image = image.to(args.device)
        outputs = pocket.ops.relocate_to_cpu(model([image]))

        size = torch.as_tensor(target["size"])
        detections = postprocessors(outputs, size.unsqueeze(0))
        scores, labels, boxes = detections[0].values()
        keep = torch.nonzero(scores >= args.thresh).squeeze(1)
        scores = scores[keep].tolist()
        labels = labels[keep].tolist()
        boxes = boxes[keep].tolist()

        with open(os.path.join(args.output_dir, dataset.filename(idx).replace('jpg', 'json')), 'w') as f:
            json.dump(dict(
                boxes=boxes, scores=scores,
                labels=labels, size=size.tolist()
            ), f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

    parser.add_argument('--lr_backbone', default=2e-6, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')

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
    parser.add_argument('--data_root', default="..", type=str)
    parser.add_argument('--pretrained', default='', help="load pretrained model", type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='./cached_detections',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--use_checkpoint", default=False, action="store_true")

    parser.add_argument("--thresh", default=.1, type=float, help="Threshold on detection scores.")

    args = parser.parse_args()
    print(args)
    main(args)
