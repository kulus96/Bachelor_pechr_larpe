#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "utilities_cpp.h"
#include "DetectorControl.h"
#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <fstream>

extern "C" void scale_img()
{
    cv::Mat img = cv::imread("test.jpg",cv::IMREAD_COLOR);
    cv::Rect2f roi = cv::Rect2f(434,266,40,90);
 
    cv::Mat img_resize, img_resize2;;
    cv::resize(img,img_resize,cv::Size(608,608));
    cv::rectangle(img_resize,roi,cv::Scalar(0,255,0),1,8);

    cv::namedWindow(std::to_string(img_resize.cols) + ", " + std::to_string(img_resize.rows), cv::WINDOW_AUTOSIZE);
    cv::imshow(std::to_string(img_resize.cols)+", "+std::to_string(img_resize.rows), img_resize);
    cv::namedWindow(std::to_string(img_resize.cols) + ", " + std::to_string(img_resize.rows)+"_roi", cv::WINDOW_AUTOSIZE);
    cv::imshow(std::to_string(img_resize.cols)+", "+std::to_string(img_resize.rows)+"_roi", img_resize(roi));

    cv::resize(img_resize,img_resize2,cv::Size(640,480));
    
    //Finding the ratio
    float x_old = img_resize.cols;
    float x_new = img_resize2.cols;
    float y_old = img_resize.rows;
    float y_new = img_resize2.rows;
    
    float rx = x_new/x_old;
    float ry = y_new/y_old;
    roi.x = roi.x*rx; roi.width = roi.width * rx;
    roi.y = roi.y*ry; roi.height = roi.height * ry;
    
    cv::rectangle(img_resize2,roi,cv::Scalar(0,0,255),1,8);
    cv::namedWindow(std::to_string(img_resize2.cols) + ", " + std::to_string(img_resize2.rows), cv::WINDOW_AUTOSIZE);
    cv::imshow(std::to_string(img_resize2.cols) + ", " + std::to_string(img_resize2.rows), img_resize2);
    cv::namedWindow(std::to_string(img_resize2.cols) + ", " + std::to_string(img_resize2.rows) + "_roi", cv::WINDOW_AUTOSIZE);
    cv::imshow(std::to_string(img_resize2.cols) + ", " + std::to_string(img_resize2.rows) + "_roi", img_resize2(roi));

    cv::rectangle(img,roi,cv::Scalar(0,0,255),1,8);
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", img);
    cv::namedWindow("Original_roi", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original_roi", img(roi));
    cv::waitKey(0);
}