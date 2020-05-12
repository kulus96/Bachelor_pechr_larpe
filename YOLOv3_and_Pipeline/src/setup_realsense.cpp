#include "setup_realsense.h"

cv::VideoCapture cap;

int setup_RealSenseCam()
{
    list *options = read_data_cfg("cfg/test1-yolov3.cfg");
    int width = option_find_int(options,"width",0);
    int height = option_find_int(options,"height",0);
    std::cout << "Preparing camera.. " << std::endl;
    
    int deviceID = 0; //set cam id
    cap.open(deviceID);
    if(!cap.isOpened())
    { 
        std::cout << "not valid camera index!" << std::endl;
        return 0;
    }
    std::cout << "Camera ready." << std::endl;
    std::cout << "width: " << width << ", height: " << height << ", frames: " << 30 << std::endl;

    return 1;
}

cv::Mat get_frame()
{
    cv::Mat frame;
    if(cap.isOpened())
    {
        while(frame.empty())
        {
            cap.read(frame);
        }
    }
    else
    {
        std::cout << "video capture is not opened correct" << std::endl;
    }
    return frame;
}