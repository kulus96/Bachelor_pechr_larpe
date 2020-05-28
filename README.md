# Bachelor_pechr_larpe

The repository contains of a pipeline implementing a YOLOv3, an AAE and a depth estimation method. YOLOv3 is a fork from https://github.com/AlexeyAB/darknet with a few insignificant changes, and AAE is from https://github.com/DLR-RM/AugmentedAutoencoder. Furthermore it contains a proposed program to generate syntetic images from CAD models.
The repository is part of the bachelor thesis made in 2020 on University of Southern Denmark called:
***Object Classification and Pose Estimation using Convolutional Neural Networks to Enable Robot Manipulation***

The repository does not include the weights trained for the objects nor does it include the videofiles used for evaluating the depth method and pipeline.

## The directories are as follows:

**Test-AAE-Program:**
- Test program of the AAE.

**YOLO-Training-Data:**
- The scripts used to generate the synthetic images.

**YOLOv3_and_Pipeline:**
- Implementation of the proposed pipeline.
- Training and evaluation methods as described in https://github.com/AlexeyAB/darknet
- Evaluation of the Depth estimation method
