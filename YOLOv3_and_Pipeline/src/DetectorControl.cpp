#include "DetectorControl.h"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "utilities_cpp.h"
#include <iostream>
#include <vector>
#include <map>
#include <bits/stdtr1c++.h>
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "darknet.h"
#include <vector>
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif


//extern void run_detector(int argc, char **argv);
int *gpus = 0;
int gpu = 0;
int ngpus = 0;

void setup(char * gpu_list)
{
	
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
}
#ifdef OPENCV

image mat_to_image(cv::Mat &mat)
{
    cv::Mat tmp;
    cv::cvtColor(mat,tmp,cv::COLOR_BGR2RGB);
    int w = tmp.cols;
    int h = tmp.rows;
    int c = tmp.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)tmp.data;
    int step = tmp.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
                //im.data[k*w*h + y*w + x] = val / 255.0f;

                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}
static network net;
int letterbox;
image **alphabet;
char *input;
float thresh_saved, hier_thresh_saved;
char **names;
int ext_out;
int dontshow;
void setupDetectorPy(char *datacfg, char *cfgfile, char *weightfile, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    letterbox = letter_box;
    thresh_saved = thresh;
    ext_out = ext_output;
    dontshow = dont_show;
    hier_thresh_saved = hier_thresh;
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    alphabet = load_alphabet();
    net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1

    if (weightfile) {
        load_weights(&net, weightfile);
    }

    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
          error("fopen failed");
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    
}
detection *dets;
int nboxes;
int numDetections = 0;
int runDetectorPy(unsigned char * im_cv)
{
    int j;
    float nms = .45; 

    //cv::waitKey(0);
    cv::Mat frame(cv::Size(608,608),CV_8UC3,im_cv);
    image im = mat_to_image(frame);


    image sized;
    if(letterbox) sized = letterbox_image(im, net.w, net.h);
    else sized = resize_image(im, net.w, net.h);
    layer l = net.layers[net.n - 1];

    //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
    //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

    float *X = sized.data;

    //time= what_time_is_it_now();
    double time = get_time_point();
    network_predict(net, X);
    //network_predict_image(&net, im); letterbox = 1;
    //printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
    //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

    nboxes = 0;
    dets = get_network_boxes(&net, im.w, im.h, thresh_saved, hier_thresh_saved, 0, 1, &nboxes, letterbox);
    if (nms) {
        if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
    }
    
    draw_detections_v3(im, dets, nboxes, thresh_saved, names, alphabet, l.classes, ext_out);
    std::string tmpName = "predictions_Yolo_" + std::to_string(numDetections);
    const char* nameImage = tmpName.c_str();
    save_image(im, nameImage);
    numDetections++;
    if (!dontshow) {
        show_image(im, "predictions_Yolo");
        
    }
    free_image(im);
    free_image(sized);

    return nboxes;
}

void endDetectorPy()
{
    free_detections(dets, nboxes);

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    //free_list_contents_kvp(options);
    //free_list(options);

    int i;
    const int nsize = 8;
    for (int j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
    std::cout << "Memory freed. Program terminated." << std::endl;
}

extern "C" float * get_bb(int numofDetections, int imgNr, float im_size, float imReal_width, float imReal_height)
{
    int numOfElements = 7;
    box b;
    int imageNr = imgNr;
    double image_size = im_size;
    double image_original_width = imReal_width;
    double image_original_height = imReal_height;
    double delta_width = image_original_width/image_size;
    double delta_height = image_original_height/image_size; 

    float *matrixx = new float[numofDetections*numOfElements];//(int *)malloc(numofDetections*numOfElements*sizeof(int)); //allocate data to matrix
    int elements = 0;
    for(int i = 0; i < numofDetections; ++i)
    {
        *(matrixx+i*numOfElements+0)  = -1;
        *(matrixx+i*numOfElements+1)  = -1;
        *(matrixx+i*numOfElements+2)  = -1;
        *(matrixx+i*numOfElements+3)  = -1;
        *(matrixx+i*numOfElements+4)  = -1;
        *(matrixx+i*numOfElements+5)  = -1;
        *(matrixx+i*numOfElements+6)  = -1;

        for(int j = 0; j < dets[i].classes; ++j)
        {            
            if(dets[i].prob[j]>=thresh_saved)
            {
                b = dets[i].bbox;
                *(matrixx+i*numOfElements+0)  = j; //class
                *(matrixx+i*numOfElements+1)  = (b.x*image_size*delta_width)/image_original_width;
                *(matrixx+i*numOfElements+2)  = (b.y*image_size*delta_height)/image_original_height;
                //*(matrixx+i*numOfElements+3)  = ((b.w-0.05)*image_size*delta_width)/image_original_width;
                //*(matrixx+i*numOfElements+4)  = ((b.h-0.05)*image_size*delta_height)/image_original_height;
                *(matrixx+i*numOfElements+3)  = ((b.w)*image_size*delta_width)/image_original_width;
                *(matrixx+i*numOfElements+4)  = ((b.h)*image_size*delta_height)/image_original_height;
                *(matrixx+i*numOfElements+5)  = imageNr; 
                *(matrixx+i*numOfElements+6)  = elements;

                elements++;
                
                continue;
            }
            
        }
    }  
    //std::cout << "Detected elements: " << elements << std::endl;
    
    return matrixx;
}
#endif
// class, width, height, dist to obj, diagonal
std::vector<std::vector<float>> obj_setup;
float pixel2mm;
float focal_length;
float sensor_diag_mm, sensor_diag_pix;
extern "C" void load_objects()
{
    char *object_cfg = "objects_cfg.txt";
    std::string class_specific;
    std::string width_str = "width_";
    std::string height_str = "height_";
    std::string intermidiate;
    list *options = read_data_cfg(object_cfg);
    int classes = option_find_int(options,"classes",0);
    float width, height, diagonal;
    focal_length = 1.93; //mm found in datasheet for realsense D435
    sensor_diag_mm = sqrt(2.7288*2.7288+1.5498*1.5498); //mm found in datasheet for OV2740
    for(int i = 0; i < classes; i++)
    {
        class_specific = std::to_string(i);
        intermidiate = width_str+class_specific;
        char *width_char = str_to_char(intermidiate);
        intermidiate = height_str+class_specific;
        char *height_char = str_to_char(intermidiate);
        width = option_find_float(options,width_char,0.0);
        height = option_find_float(options,height_char,0.0);
        diagonal = sqrt(width*width+height*height);

        obj_setup.push_back(std::vector<float>());
        obj_setup[i].push_back((float)i);
        obj_setup[i].push_back(width);
        obj_setup[i].push_back(height);
        obj_setup[i].push_back(diagonal);
    }
    float tmp = option_find_float(options,"dist_to_objects",0.0);
}
extern "C" float get_depth(float* dets, float im_width, float im_height)
{
 //how to take rotation into account? rotation will affect the size of the bbox
//requires the AE
    //list *objects = read_data_cfg("object_cfg.cfg");
    float depth = 0;
    float width = dets[3]*im_width;
    float height = dets[4]*im_height;
    sensor_diag_pix = sqrt(im_width*im_width+im_height*im_height);

    float dets_diag_mm = (sensor_diag_mm*sqrt(width*width+height*height))/sensor_diag_pix;
    //obj_setup: class, width, height, diagonal
    //dets: class, x, y, width, height, image number, detection element
    //std::cout << obj_setup[dets[0]][3] << "*" << focal_length << "*" << dets_diag_mm << std::endl;
    // dist = (x/x')*f    - pinhole model
    depth = obj_setup[dets[0]][3]*focal_length/dets_diag_mm;
    //missing conversion from pixel to mm
    //std::cout << "estimated depth in mm: " << depth << std::endl;
    
    return depth;
}




