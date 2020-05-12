import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser
import random
import math
import shutil
import numpy.ctypeslib as ctl 
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
from auto_pose.ae import factory, utils
import Realsense_Camera as cam

def get_Translation2d(arr):
    pose = []
    im_width = 640
    im_height = 480

    for row in range(len(arr)): 
        pose.append([arr[row][5], arr[row][6], arr[row][0],arr[row][1]*im_width, arr[row][2]*im_height]) #imageNr, detectionNr, class, x, y

    return pose

if __name__ == '__main__':

#setup of YOLOv3 library:
    libname = 'libdarknet.so'
    libdir = './'
    lib=ctl.load_library(libname,libdir)
    #setup of YOLOv3 functions:
    Yolo_setup = lib.run_detector
    Yolo_run = lib.runDetectorPy
    Yolo_getbb = lib.get_bb
    Yolo_end = lib.endDetectorPy
    Yolo_loadObj = lib.load_objects
    Yolo_getDepth = lib.get_depth
    #set arguments and return types
    Yolo_setup.argtypes = [ctypes.c_int, POINTER(c_char_p)]
    Yolo_getbb.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]    
    Yolo_run.argtypes = [ctypes.c_char_p]
    Yolo_getDepth.argtypes = [POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float]

    Yolo_run.restype = ctypes.c_int
    Yolo_getDepth.restype = ctypes.c_float

    #set arguments
    argv = 8 
    argc = (c_char_p*argv)('./darknet', 'detector','runYolov3','Batchelor.data', 'CAD_MODEL.cfg', 'yoloweights/Real_model_best.weights', '-dont_show') #5 test files: .data .cfg .weight img
    
    Yolo_setup(argv,argc) #Setup of Yolo
    Yolo_loadObj() #load figure parameters for depth estimation

    #camera setup
    Livefootage = 0 # set to 0 if video is used
    video_path = 'Depth_fil_Test2.bag'
    camera = cam.Realsense_Camera(640,480,video_path,Livefootage)

    area = 5 #Kernel for averaging of size (2*area + 1)x(2*area +1)
    depth_file = open("depth_file_area_" + str(area) + "_Test3" + ".txt",'a')

    pressed = 0
    numImg = -1
    numTest = 0
    #pressed != 116 or

    while cam.Realsense_Check_Video_ended(camera): # 116 = t
        
        #cv2.waitKey(0)
        img = cam.Realsense_Frame_Color(camera)
        cam.Realsense_Frame_Depth(camera)
        cam.Realsense_Show_Depth(camera)

        if len(img) < 3 or camera.depth_image.shape[0] == 1:
            continue
        
        #MAT translation to char array
        img_res = cv2.resize(img,(608,608))
        frameArray1_tmp = np.fromstring(img_res,np.uint8)
        frameArray2_tmp = np.reshape(frameArray1_tmp,(608,608,3))
        framearray = frameArray2_tmp.tostring()

        img_width, img_height, _ = img.shape
    
        numImg += 1 # image number! needs to be incremented for each frame
        

        numDet = Yolo_run(framearray) #run yolo return number of detections
        if numDet > 0:
            Yolo_getbb.restype = ndpointer(dtype=ctypes.c_float, shape=(numDet,7)) #allocate space in memory for detection matrix

            det_matrix = Yolo_getbb(numDet, numImg, 608, img_width, img_height) #returns detection matrix using format:
            #class, x, y, width, height, image number, Detection element
            pose_tmp = []
            for rows in det_matrix:
                if rows[0] != -1:
                    pose_tmp.append(rows)

            pose = get_Translation2d(pose_tmp) #find x and y for depth file 
                    
            for i in range(len(pose_tmp)):
                pose_tmp2 = (ctypes.c_float*len(pose_tmp[i]))(*pose_tmp[i])
                depth_file.write(str(numImg) + ",") # frame number
                depth_file.write(str(pose[i][2]) + ",") #class of object
                depth_file.write(str(Yolo_getDepth(pose_tmp2, img_width, img_height)/1000) + ",") #depth estimation in meters
                depth_file.write(str(cam.Realsense_get_Distance(camera, int(pose[i][3]),int(pose[i][4]), area,1)) + "\n") #camera depth 
            
        cv2.imshow("show image",img)
        pressed = cv2.waitKey(1)
    depth_file.close()
    Yolo_end() 
    cam.Realsense_Terminate(camera)
