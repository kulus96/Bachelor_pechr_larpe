# Object Classification and Pose Estimation using Convolutional Neural Networks to Enable Robot Manipulation

The repository contains of a pipeline implementing a YOLOv3, an AAE and a depth estimation method. YOLOv3 is a fork from https://github.com/AlexeyAB/darknet with a few insignificant changes, and AAE is from https://github.com/DLR-RM/AugmentedAutoencoder. Furthermore it contains a proposed program to generate syntetic images from CAD models.

The repository is part of a bachelor thesis for Robot Systems Engineering made in the spring of 2020 on University of Southern Denmark.

The repository does not include the weights trained for the objects nor does it include the videofiles used for evaluating the depth method and pipeline.

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
