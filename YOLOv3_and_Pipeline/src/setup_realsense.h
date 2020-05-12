#ifndef SETUP_REALSENSE_H
#define SETUP_REALSENSE_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "darknet.h"
#include "option_list.h"

int setup_RealSenseCam();
cv::Mat get_frame();


#endif