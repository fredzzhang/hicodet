# Detection Utilities

## Train and test DETR on HICO-DET

To train the DETR model with ResNet50 as the backbone, run the following command.
```bash
python main_detr.py --world_size 8 --epochs 200 --lr_drop 150 &>out &
```
This starts from a randomly initialised model. To start from a MS COCO pretrained model, first download the checkpoint by executing `bash download_checkpoint.sh`, then run the following command.

```bash
python main_detr.py --world_size 8 --epochs 50 --lr_drop 30 --pretrained checkpoints/detr-r50-e632da11.pth &>out &
```
To test a model a pre-trained model, use the flag `--pretrained` to specify the path. To test a model trained using this repo, use the flag `--resume` to specify the path. If you use both flags, the pre-trained model will be overriden by the newly trained model.
```bash
python main_detr.py --eval --partition test2015 --pretrained /path/to/checkpoint --resume /path/to/checkpoint
```
For more options regarding the customisation of network architecture and hyper-parameters, run `python main_detr.py --help` to find out. Alternatively, refer to the [original repo](https://github.com/facebookresearch/detr).

## Generate detections using Faster R-CNN

```bash
python preprocessing.py --partition train2015
```

A Faster R-CNN model pre-trained on MS COCO will be used by default to generate detections. Use the argument `--partition` to specify the subset to run the detector on. To run a Faster R-CNN model with fine-tuned weights, use the argument `--ckpt-path` to load the model from specified checkpoint. Run `python preprocessing.py --help` to find out more about post-processing options. The generated detections will be saved in a directory named after the parition e.g. `train2015`.

## Generate ground truth detections

```bash
python generate_gt_detections.py --partition test2015
```

Generate detections from the ground truth boxes. Notes that since the ground truth is formatted as box pairs, when the same human (or object) instance appears multiple times in different pairs, they will be saved multiple times. Moreover, the same instance could have slightly different annotated boxes when appearing in different pairs. For this reason, we do not perform NMS in the code and leave the choice of post-processing to the users. The generated detections will be saved in a directory named after the parition e.g. `test2015_gt`. 

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

Start from the pre-trained detector on MS COCO and fine-tune the detector on HICO-DET. Note that setting the environmental variable `CUDA_VISIBLE_DEVICES` is necessary and should __NOT__ be omitted (Refer to [#7](https://github.com/fredzzhang/hicodet/issues/7)). Run `python faster_rcnn.py --help` for more options regarding the hyper-parameters.