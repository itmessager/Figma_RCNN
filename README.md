# Figma RCNN(Fine-grained Multi-Attribute RCNN)
Person Detection and Multi-Attributes Recognition with only one Jointly-Trained Holistic CNN Model<br/>
The master branch works with tensorpack 0.9 <br/>
It is a part of the Figma RCNN project developed by Junlin Gu, Graduate student at UESTC.

## Main References
+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Human Attribute Recognition by Deep Hierarchical Contexts](http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html)
+ [An All-In-One Convolutional Neural Network for Face Analysis](https://www.researchgate.net/publication/309663347_An_All-In-One_Convolutional_Neural_Network_for_Face_Analysis)

## Dependencies
+ Python 3.3+; OpenCV
+ TensorFlow ≥ 1.6
+ Tensorpack ≥ 0.9
+ pycocotools: `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/FasterRCNN/)
  from tensorpack model zoo
+ [COCO data](http://cocodataset.org/#download). It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train201?.json
    instances_val201?.json
  train201?/
    COCO_train201?_*.jpg
  val201?/
    COCO_val201?_*.jpg
```
+ [Wider Attibutes data](https://drive.google.com/open?id=0B-PXtfvNMLanWEVCaHZnR0RHSlE). It needs to have the following directory structure:
```
wider attibutes/
  Anno/
    wider_attribute_trainval.json
    wider_attribute_test.json
  train/
    0--Parade/
      0_Parade_marchingband_?.jpg

    1--Handshaking/
      1_Handshaking_Handshaking_?.jpg
    ...
  test/
    0--Parade/
      0_Parade_marchingband_?.jpg
    1--Handshaking/
      1_Handshaking_Handshaking_?.jpg
    ...
```

## Usage
### Installation

Setup Docker Environment on Ubuntu host(recommended)<br/>
Download the image file and install the Docker environment, please click [here](https://blog.csdn.net/weixin_38502181/article/details/84632610) to view my blog

### Train:

To train on a single machine:
```
./attr_train.py --config \
    BACKBONE.WEIGHTS=/path/to/COCO-R50C4-MaskRCNN-Standard.npz
```

Options can be changed by either the command line or the `tensorpack_config.py` file (recommended).

### Inference:

To predict on an image (needs DISPLAY to show the outputs):
```
./demo_cam.py 
--image
/path/to/input.jpg
--cam
0
--obj_model
all-in-one
--obj_ckpt
/root/to/checkpoint
--obj_config
DATA.BASEDIR=/path/to/COCO/DIR
```

The trained models can be downloaded in the [Baidu Cloud] (Waiting upload).

## Results

The models' detection branch are trained on COCO trainval35k and evaluated on COCO minival2014 using mAP@IoU=0.50:0.95. attributes branch are trained on Wider Attribute trainval and evaluated on Wider Attribute test using mAP.
The models are fine-tuned from ResNet pre-trained R50C4 models in
[tensorpack model zoo](http://models.tensorpack.com/FasterRCNN/)

Performance in [Person Detection](https://github.com/facebookresearch/Detectron/) can
be approximately reproduced.

    | Backbone  | mAP (box) | Detectron mAP (box) |Configurations                            |
    | R50-C4    | 33.3      | 32.8                |TRAIN.LR_SCHEDULE=[120000, 240000, 280000]|

Performance in [Person Attributes Recognition](http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html) can
be approximately reproduced.
   
    |      Atrributes   | AP (positive/negative)     |
    |        male       |        0.9503              |
    |      longhair     |        0.8598              |
    |      sunglass     |        0.7342              |
    |        hat        |        0.9477              |
    |       tshirt      |        0.7892              |
    |     longsleeve    |        0.9602              |
    |       formal      |        0.8084              |
    |       shorts      |        0.9105              |
    |       jeans       |        0.7461              |
    |     longpants     |        0.9711              |
    |       skirt       |        0.8454              |
    |     facemask      |        0.7372              |
    |       logo        |        0.8912              |
    |      stripe       |        0.6159              |
    |        mAP        |        0.8405              |
 
## Some examples

Here are some visualization results of the figma rcnn model.<br/>
![Image text](https://github.com/itmessager/Figma_RCNN/blob/master/demo/1.png)<br/>
![Image text](https://github.com/itmessager/Figma_RCNN/blob/master/demo/2.png)<br/>
![Image text](https://github.com/itmessager/Figma_RCNN/blob/master/demo/3.png)<br/>
![Image text](https://github.com/itmessager/Figma_RCNN/blob/master/demo/4.png)<br/>
![Image text](https://github.com/itmessager/Figma_RCNN/blob/master/demo/5.png)