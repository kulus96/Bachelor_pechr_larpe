#ifndef YOLOV3CONTROL_H
#define YOLOV3CONTROL_H

#include <stdio.h>
#include "Log.h"
#include "../include/darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"




#ifdef __cplusplus
extern "C" {
#endif
    
    void getROI(image img, detection *dets, int numOfDetections, float thresh, int save);
    //void runYolov3(char* datacfg, char *cfg, char *weights, float thresh, float hier_thresh, int dont_show, int ext_output, char *outfile, int letter_box, int benchmark_layers);
    float estimate_depth(detection *dets, float thresh);
    void setup_objects();
    void getTest_AAE();
    void Rundemo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers);
    //void runLiveYolo(char* datacfg, char *cfg, char *weights, float thresh, float hier_thresh, int dont_show, int ext_output, char *outfile, int letter_box, int benchmark_layers);
    //void TrainYolov3_Diff(char* weightfile, char* cfg,int *gpus, int ngpus, int clear, int dont_show, int calc_map, int mjpeg_port, int show_imgs, int benchmark_layers);
#ifdef __cplusplus
}
#endif
  

#endif 