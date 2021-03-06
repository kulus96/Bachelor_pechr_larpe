# Object Classification and Pose Estimation using Convolutional Neural Networks to Enable Robot Manipulation

The repository contains of a pipeline implementing a YOLOv3, an AAE and a depth estimation method. YOLOv3 is a fork from https://github.com/AlexeyAB/darknet with a few insignificant changes, and AAE is from https://github.com/DLR-RM/AugmentedAutoencoder. Furthermore it contains a proposed program to generate syntetic images from CAD models.

The repository is part of a bachelor thesis for Robot Systems Engineering made in the spring of 2020 on University of Southern Denmark. The project is made by Lars Pedersen and Peter Christiansen.

The repository does not include the weights trained for the objects nor does it include the videofiles used for evaluating the depth method and pipeline.
## Abstract

We propose a pipeline, which returns 6D pose information of objects given an RGB image. The pipeline is based on YOLOv3 for object detections, three AAEs for rotation estimation and a depth estimation method based on the pinhole model principles. We examine the effects on the performance of YOLOv3 when trained on synthetic CAD generated images and a mixture of real and synthetic images to potentially eliminate the need for manually annotating training sets for CNNs.

The methods are evaluated using three different objects. YOLOv3 is evaluated to a mAP of 41.01% on weights trained on synthetic images and a mAP of 99.21% on the mixture of CAD and real training images. The detector is further evaluated in a environment with the objects placed in a cluttered manner achieving a mAP of 20.87% and 93.92% for the synthetic and the mixed trained weights respectively.
The AAEs are evaluated to an average of 50.5%, 81.5% and 62.5% across the three AAEs. 
The depth estimation method is evaluated to a precision of +- 5.38% in the best case and +- 14.76% in the worst case. The depth estimation is sensitive to rotation of the objects finding an increase in error of 981.7\%, when an object is rotated by 90 degree.

## Contents

**Test-AAE-Program:**
- Test program of the AAE with [instructions](Test-AAE-Program/ReadMe).

**YOLO-Training-Data:**
- The scripts used to generate the synthetic images with [instructions](YOLO-Training-Data/ReadMe).

**YOLOv3_and_Pipeline:**
- Implementation of the proposed [pipeline](YOLOv3_and_Pipeline/finalpipeline_one.py).
- Implementation of the proposed [depth estimation method](YOLOv3_and_Pipeline/src/DetectorControl.cpp)
- Training and evaluation methods of YOLOv3 as described [here](https://github.com/AlexeyAB/darknet)
- Evaluation of the [Depth estimation method](YOLOv3_and_Pipeline/Depth_Test.py)

## Requirements

1. Have YOLOv3 installed
2. Have AAE installed

When the AAE is installed the directory must be placed inside the YOLOv3 directory. 

## How to

**[in Pipeline:](YOLOv3_and_Pipeline/finalpipeline_one.py)**
- To show the AAE predictions while running, set the `Show` to 1 
- To show the predictions made by YOLOv3, set the remove the `-dont_show` command from `argc` and change `argv` to 8


