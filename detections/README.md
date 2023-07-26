# Detection Utilities

## Train and test DETR on HICO-DET

To fine-tune [DETR](https://arxiv.org/abs/2005.12872) from a MS COCO pretrained model, first download the checkpoints from the table below. The following command is an example for fine-tuning DETR-R50.

```bash
python main_detr.py --world_size 8 --epochs 30 --pretrained checkpoints/detr-r50-e632da11.pth &>out &
```
To test a pre-trained model, use the flag `--pretrained` to specify the path. To test a model trained using this repo, use the flag `--resume` to specify the path. If you use both flags, the pre-trained model will be overridden by the newly trained model.
```bash
python main_detr.py --eval --partition test2015 --pretrained /path/to/checkpoint --resume /path/to/checkpoint
```
For more options regarding the customisation of network architecture and hyperparameters, run `python main_detr.py --help` to find out. Alternatively, refer to the [original repo](https://github.com/facebookresearch/detr). For convenience, we provide fine-tuned DETR weights below.

|Model|mAP|mRec|HICO-DET|Size|Inference|MS COCO|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
|DETR-R50|`50.60`|`72.36`|[weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)|`159MB`|`0.036s`|[weights](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)|
|DETR-R101|`51.68`|`73.20`|[weights](https://drive.google.com/file/d/1pZrRp8Qcs5FNM9CJsWzVxwzU7J8C-t8f/view?usp=sharing)|`232MB`|`0.050s`|[weights](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth)|
|DETR-R101-DC5|`52.38`|`74.40`|[weights](https://drive.google.com/file/d/1kkyVeoUGb8rT9b5J5Q3f51OFmm4Z73UD/view?usp=sharing)|`232MB`|`0.097s`|[weights](https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth)|

## Train and test advanced variants of DETR on HICO-DET

First build the CUDA operators for `MultiScaleDeformableAttention`, as follows

```bash
cd h_detr/models/ops
# Append flag --user to the command below if you are using compute clusters
# and do not have write permission in system directories.
python setup.py build install
```

To fine-tune an advanced variant of DETR, download the MS COCO pre-trained weights from the table below. The following commands are examples for fine-tuning the [Deformable DETR](https://arxiv.org/abs/2010.04159) with [additional techniques](https://arxiv.org/abs/2203.03605) and the more advanced H variants ([hybrid matching](https://arxiv.org/abs/2207.13080)). If the GPUs you use do not have sufficient memory, use the `--use_checkpoint` option to [save memory](https://pytorch.org/docs/stable/checkpoint.html).

```bash
# Train Deformable DETR-R50 with additional techniques
python main_h_detr.py --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                      --k_one2many 0  --epochs 30 --world_size 8 --batch_size 2 \
                      --pretrained /path/to/checkpoint

# Train H-Deformable DETR-R50 with additional techniques
python main_h_detr.py --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                      --num_queries_one2many 1500 --k_one2many 6 \
                      --epochs 30 --world_size 8 --batch_size 2 \
                      --pretrained /path/to/checkpoint

# Train H-Deformable DETR-SwinL with additional techniques
python main_h_detr.py --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                      --backbone swin_large --drop_path_rate 0.5 \
                      --num_queries_one2one 900 --num_queries_one2many 1500 --k_one2many 6 \
                      --epochs 30 --world_size 8 --batch_size 2 --weight_decay 0.05 \
                      --pretrained /path/to/checkpoint
```

To test a MS COCO pre-trained model, use the flag `--pretrained` to specify the path. To test a model fine-tuned using this repo, use the flag `--resume` to specify the path. If you use both flags, the pre-trained model will be overridden.

```bash
# Test MS COCO pre-trained Deformable DETR-R50
python main_h_detr.py --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                      --world_size 1 --batch_size 1 --eval \
                      --pretrained /path/to/checkpoint

# Test HICO-DET fine-tuned Deformable DETR-R50
python main_h_detr.py --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                      --world_size 1 --batch_size 1 --eval \
                      --resume defm-detr-r50-dp0-mqs-lft-iter-2stg-hicodet.pth

```

|Model|mAP|mRec|HICO-DET|MS COCO|
|:-|:-:|:-:|:-:|:-:|
|Defm-DETR-R50|`53.30`|`77.86`|[weights](https://drive.google.com/file/d/1A0FQQLLQE32j7YISHsJZO76dK9vqy1ll/view?usp=share_link)|[weights](https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth)|
|H-Defm-DETR-R50|`54.16`|`78.39`|[weights](https://drive.google.com/file/d/1cwMJNMQALDrVdTxQL6Vdw66thpgeyq-2/view?usp=share_link)|[weights](https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth)|
|H-Defm-DETR-SwinL|`64.23`|`85.17`|[weights](https://drive.google.com/file/d/1wge-CC1Fx67EHOSXyHGHvrqvMva2jEkr/view?usp=share_link)|[weights](https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/decay0.05_drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth)|

## Generate detections using the H-DETR variants

```bash
# Use MS COCO pre-trained H-Deformable DETR-R50
python h_detr_cache.py  --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                        --num_queries_one2many 1500 \
                        --pretrained /path/to/checkpoint

# Use HICO-DET fine-tuned H-Deformable DETR-R50
python h_detr_cache.py  --with_box_refine --two_stage --mixed_selection --look_forward_twice \
                        --num_queries_one2many 1500 \
                        --resume h-defm-detr-r50-dp0-mqs-lft-iter-2stg-hicodet.pth
```
By default, detections will be cached under the directory `cached_detections` in the form of `.json` files. To cache detections for custom datasets, refer to the instructions in the script.

## Generate detections using Faster R-CNN

```bash
python preprocessing.py --partition train2015
```

A Faster R-CNN model pretrained on MS COCO will be used by default to generate detections. Use the argument `--partition` to specify the subset to run the detector on. To run a Faster R-CNN model with fine-tuned weights, use the argument `--ckpt-path` to load the model from specified checkpoint. Run `python preprocessing.py --help` to find out more about post-processing options. The generated detections will be saved in a directory named after the partition e.g. `train2015`.

## Generate ground truth detections

```bash
python generate_gt_detections.py --partition test2015
```

Generate detections from the ground truth boxes. Notes that since the ground truth is formatted as box pairs, when the same human (or object) instance appears multiple times in different pairs, they will be saved multiple times. Moreover, the same instance could have slightly different annotated boxes when appearing in different pairs. For this reason, we do not perform NMS in the code and leave the choice of post-processing to the users. The generated detections will be saved in a directory named after the partition e.g. `test2015_gt`. 

## Visualise detections

```bash
python visualise.py --detection-root ./test2015 --partition test2015 --image-idx 3000
Image name:  HICO_test2015_00003102.jpg
```

Visualise detections for an image. Use argument `--detection-root` to specify the directory of generated detection files and `--partition` to specify the subset. To select a specific image, use the argument `--image-idx`. The name of corresponding image file will also be printed.

## Evaluate detections

```bash
python eval_detections.py --detection-root ./test2015
```

Evaluate the mAP of the detections against the ground truth object detections of HICO-DET. Use the argument `--partition` to specify the subset to evaluate against. The default is `test2015`. Use the argument `--detection-root` to point to the directory where detection results are saved. Note that due to the multiple appearances of the same instance in different pairs, NMS will also be applied on the ground truth detections. This could cause some unforeseeable issues. Therefore, this evaluation is somewhat coarse and should only be used a diagnostic tool. Run `python eval_detections.py --help` to find out more about post-processing options.

## Fine-tune Faster R-CNN on HICO-DET

```bash
CUDA_VISIBLE_DEVICES=0 python train_faster_rcnn.py
```

Start from the pre-trained detector on MS COCO and fine-tune the detector on HICO-DET. Note that setting the environmental variable `CUDA_VISIBLE_DEVICES` is necessary and should __NOT__ be omitted (Refer to [#7](https://github.com/fredzzhang/hicodet/issues/7)). Run `python faster_rcnn.py --help` for more options regarding the hyperparameters.
