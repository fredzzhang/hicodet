# HICO-DET
Utilities for the human-object interaction detection dataset [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/)

## Supported Utilities

- [x] __NEW!__ [Train and test DETR on HICO-DET](https://github.com/fredzzhang/hicodet/tree/main/detections#train-and-test-detr-on-hico-det)
- [x] [A command-line style dataset navigator](https://github.com/fredzzhang/hicodet/tree/main/utilities#dataset-navigator)
- [x] [Large-scale visualisation in web page](https://github.com/fredzzhang/hicodet/tree/main/utilities#generate-and-visaulise-box-pairs-in-large-scales)
- [x] [Generate object detections with Faster R-CNN](https://github.com/fredzzhang/hicodet/tree/main/detections#generate-detections-using-faster-r-cnn)
- [x] [Generate ground truth object detections](https://github.com/fredzzhang/hicodet/tree/main/detections#generate-ground-truth-detections)
- [x] [Visualise detected objects](https://github.com/fredzzhang/hicodet/tree/main/detections#visualise-detections)
- [x] [Evaluate object detections](https://github.com/fredzzhang/hicodet/tree/main/detections#evaluate-detections)
- [x] [Fine-tune Faster R-CNN on HICO-DET](https://github.com/fredzzhang/hicodet/tree/main/detections#fine-tune-the-detector-on-hico-det)

## Installation Instructions
1. Download the repo with `git clone https://github.com/fredzzhang/hicodet.git`
2. Prepare the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/hicodet
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link
    ```bash
    cd /path/to/hicodet
    ln -s /path/to/hico_20160224_det ./hico_20160224_det
    ```
3. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
4. Make sure the environment you created for Pocket is activated. You are good to go!

## Dataset Class
The implementation of the dataset class can be found in `hicodet.py`. Refer to the [documentation](./DOC.md) to find out more about its usage. For convenience, the dataset class has been included in the [Pocket](https://github.com/fredzzhang/pocket) library, accessible via `pocket.data.HICODet`.

## License

[MIT License](./LICENSE)