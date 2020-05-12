#ifndef DETECTORCONTROL_H
#define DETECTORCONTROL_H

#include "image.h"
#include "image_opencv.h"


#ifdef __cplusplus
extern "C" {
#endif
void setupDetectorPy( char *datacfg, char *cfgfile, char *weightfile, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);
int runDetectorPy(unsigned char * im);
float *get_bb(int numofdetections, int imgNr, float im_size, float imReal_width, float imReal_height);
float get_depth(float *dets, float width, float height);
void load_objects();
void endDetectorPy();

#ifdef __cplusplus
}
#endif
        
#endif

