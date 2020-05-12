
#include "YoloV3Control.h"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "utilities_cpp.h"
#include <sys/time.h>
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "darknet.h"
#include "image_opencv.h"
#include "demo.h"


/*
g++ `pkg-config --cflags opencv` -o Yolov3Control Yolov3Control.cpp `pkg-config --libs opencv`
*/


std::vector<objects*> obj_vect;

cv::Mat image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) { //rows
        for (int x = 0; x < img.w; ++x) { //cols
            for (int c = 0; c < img.c; ++c) { //colour channels
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.at<cv::Vec3b>(cv::Point(x,y))[c] = val*255;
                
            }
        }
    }
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR); //image type is RGB
    return mat;
}

cv::Point2f getCenterCoordinates(box b, cv::Mat img)
{
    cv::Point2f center_point;
    int img_width = img.size().width;
    int img_height = img.size().height;
    center_point = cv::Point2f(b.x*img_width,b.y*img_height);

    return center_point;
}

cv::Rect2f bbox_to_cooardinates(box b, cv::Mat img)
{
    cv::Rect2f bounding_box;
    int img_width = img.size().width;
    int img_height = img.size().height;
    float box_height = b.h*img_height;
    float box_width = b.w*img_width;

    float x = std::abs((b.x-b.w/2)*img_width);
    float y = std::abs((b.y-b.h/2)*img_height);

    bounding_box = cv::Rect2f(x,y,box_width,box_height);
    return bounding_box;

}

extern "C" void getROI(image img, detection *dets, int numOfDetections, float thresh, int save)
{
    cv::Mat full_img = image_to_mat(img);
    std::vector<cv::Mat> roi_images;
    cv::Mat roi_img;
    box b; cv::Rect2f roi_Rect; 
    float x,y,width,height;
    std::string name;
    

    for(int i = 0; i < numOfDetections;++i) //runs through the detections
    {
        for(int j=0;j<dets[i].classes;++j) //runs through the probability of the detection being one of the classes
        {
            if(dets[i].prob[j]>=thresh)
            {
                name = std::to_string(i);
                b = dets[i].bbox;

                roi_Rect = bbox_to_cooardinates(b, full_img);
                roi_img = full_img(roi_Rect);
                
                cv::imshow(name,roi_img);
                if(save)
                {
                    std::string name = std::to_string(i)+".jpg";
                    cv::imwrite(name,roi_img);
                }
            }
        }
    }

}

/*char * str_to_char(std::string intermidiate)
{
    char* char_return = new char(intermidiate.size());
    intermidiate.copy(char_return, intermidiate.size());
    char_return[intermidiate.size()] = '\0';

    return char_return;
}

std::vector<char*> list_dir(char *path_in)
{
    FILE *fpipe;
    char c = 0;
    std::string inter_string = path_in;
    std::string intermediate;
    std::vector<char*> files_in;
    std::vector<char*> files_out;
    std::string command = "ls ";
    command.append(path_in);
    const char *test = command.c_str();    
    fpipe = (FILE*)popen(test,"r");

    while(fread(&c, sizeof(c),1,fpipe))
    {  
        
        if(c == '\n' && intermediate.size() > 4)
        {
            char *file_name = str_to_char(intermediate);
            files_in.push_back(file_name);
            intermediate.clear();
            inter_string = path_in;
            
        }
        else{
            intermediate += c; 
        }       
    }

    fclose(fpipe);
    return files_in;
}*/

void create_detection_file(detection *dets,int boxes, std::string name, float thresh)
{
    for(int i = 0; i < 3; i++) //delete jpg
    {
        name.pop_back();
    }
    name += "txt";
    std::ofstream outfile(name); // <object-class> <x> <y> <width> <height>
    
    int best_class;
    for(int j = 0; j < boxes; j++)
    {
        for(int i = 0; i < dets[j].classes; i++)
        {
            if(dets[j].prob[i] >= thresh)
            {
                best_class = i;
                float x = dets[j].bbox.x;
                float y = dets[j].bbox.y;
                float width = dets[j].bbox.w;
                float height = dets[j].bbox.h;
                outfile << best_class << " " << x << " " << y << " " << width << " " << height << std::endl;
            }
        }   
    }
    outfile.close(); 
}

extern "C" void setup_objects()
{
    char *object_cfg = "objects_cfg.txt";
    std::string class_specific;
    std::string width_str = "width_";
    std::string height_str = "height_";
    std::string intermidiate;
    list *options = read_data_cfg(object_cfg);
    int classes = option_find_int(options,"classes",0);
    std::cout << classes << std::endl;
    for(int i = 0; i < classes;i++)
    {
        class_specific = std::to_string(i);
        intermidiate = width_str+class_specific;
        char *width_char = str_to_char(intermidiate);
        intermidiate = height_str+class_specific;
        char *height_char = str_to_char(intermidiate);
        objects *c = new objects(); 
        c->class_obj = i;
        c->original_width = option_find_float(options,width_char,0.0);
        c->original_height = option_find_float(options,height_char,0.0);
        c->dist = option_find_float(options,"dist_to_objects",0.0);
        obj_vect.push_back(c);
        
    }
    std::cout << "Setup of objects complete" << std::endl;
}

extern "C" void getTest_AAE()
{
    std::vector<char*> im = list_dir("~/Documents/Vali/Validation");
    std::vector<cv::Rect2f> boxe;
    std::string path = "Validation/";
    std::ifstream dataFile(path+im[1]);
    
    //dataFile.open(im[1]);
    std::string boxes;
    std::string intermediate;
    box inter;
    cv::Rect2f detectedRoi;
    cv::Mat roiImg;
    int k;
    for(int i = 0; i < im.size(); i = i+2)
    {
        std::ifstream dataFile(path+im[i+1]);
        cv::Mat img = cv::imread(path+im[i],cv::IMREAD_COLOR);
        //cv::waitKey(0);
        
        while(std::getline(dataFile,boxes))
        {
            k = 0;
            for(std::string::iterator j=boxes.begin()+2; j != boxes.end(); j++)
            {
                if(*j != ' ' && j!=boxes.end()-1)
                {
                    intermediate += *j;   
                }
                else
                {
                  
                    if(k == 0)
                    {
                        inter.x = std::stof(intermediate);
                        //std::cout << inter.x << std::endl;
                    }
                    else if(k == 1)
                    {
                        inter.y = std::stof(intermediate);
                        //std::cout << inter.y << std::endl;
                    }
                    else if(k == 2)
                    {
                        inter.w = std::stof(intermediate);
                        //std::cout << inter.w << std::endl;
                    }
                    else if(k == 3)
                    {
                        inter.h = std::stof(intermediate);
                        //std::cout << inter.h << std::endl;
                    }
                    k++;
                    intermediate.clear();
                }
                
            }
            detectedRoi = bbox_to_cooardinates(inter,img);
            boxe.push_back(detectedRoi);
        }
        for(int j = 0; j < boxe.size();j++)
        {
            std::cout << im[i] << ": " << boxe[j].x << ", " << boxe[j].y << ", " << boxe[j].width << ", " << boxe[j].height << std::endl;
            std::string name = "imagesAE/"+std::to_string(i)+"_"+std::to_string(j)+".jpg";
            if(boxe[j].width > 0 && boxe[j].height > 0 && boxe[j].x > 0 && boxe[j].y > 0)
            {
                roiImg = img(boxe[j]);
            }
            /*else
            {
                std::cout << im[i] << ":" << boxe[j].x << ", " << boxe[j].y << ", " << boxe[j].width << ", " << boxe[j].height << std::endl;
            }*/
            
            //cv::imshow(name,roiImg);
            //cv::imwrite(name,roiImg);
        }
        boxe.clear();
        
    }

}
template <class T>
double pinhole_dist (T f, T sX, T sX_prime){
    return (sX/sX_prime) * f;
}

extern "C" float estimate_depth(detection *dets, float thresh)
{ //how to take rotation into account? rotation will affect the size of the bbox
//requires the AE
    //list *objects = read_data_cfg("object_cfg.cfg");

   setup_objects();
    double depth = 0;
    double ratio = 0;
    double focal_length = 1.93; //mm found in datasheet for realsense D435
    double pixel_to_mm = 1; //needs to be adjusted
    float object_w, object_w_prime;
    
    for(int i = 0; i < dets->classes;i++)
    {
        if(dets->prob[i]>thresh)
        {
            object_w_prime = dets->bbox.w;
            object_w = obj_vect[i]->original_width;
            depth = pinhole_dist<double>(focal_length,object_w,object_w_prime);
        }   
    }
    std::cout << "estimated depth: " << depth << std::endl;
    return depth * pixel_to_mm;
}

extern "C" void runYolov3(char* datacfg, char *cfg, char *weights, float thresh, float hier_thresh, int dont_show, int ext_output, char *outfile, int letter_box, int benchmark_layers)
{
    char *object_in = "Images/";
    std::string object_in_str = "Images/";
    std::string object_out = "detected_objects/";
    std::string intermediate;
    std::vector<char*> files_in;
    std::vector<char*> files;
    std::vector<std::string> files_out;
    files=list_dir(object_in);
    for(int i=0; i<files.size();i++)
    {
        intermediate = object_in_str + files[i];
        char * inter = str_to_char(intermediate);
        files_out.push_back(object_out+files[i]);
        files_in.push_back(inter);
    }
        
}
