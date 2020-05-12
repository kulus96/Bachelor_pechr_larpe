import multiprocessing as mp
import time 
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
    mid_x = im_width/2
    mid_y = im_height/2

    for row in range(len(arr)): 
        pose.append([arr[row][5], arr[row][6], arr[row][0],abs(arr[row][1]*im_width-mid_x), abs(arr[row][2]*im_height-mid_y)]) #imageNr, detectionNr, class, x, y

    return pose

def get_image_for_AAE(Index,BB,Received_Im):
    # Yolo format : <object-class> <x> <y> <width> <height>
    
    if not (Index == BB[0]):
        print("Bounding box class does not match with the AAE called ")
        exit(0)
    
    Size_of_Image_y,Size_of_Image_x,_ = Received_Im.shape
    
    
    Center_x = int(BB[1] * Size_of_Image_x+0.5)
    Center_y = int(BB[2] * Size_of_Image_y+0.5)
    Width = int(BB[3] * Size_of_Image_x+0.5)
    Height = int(BB[4] * Size_of_Image_y+0.5)
    
    if (Width < Height):
        tmp_img = np.zeros((Height, Height,3),np.uint8)
    else:
        tmp_img = np.zeros((Width, Width,3), np.uint8)

    bounds = np.array([int((Center_y-Height/2.0)+0.5), int((Center_y+Height/2.0)+0.5), int((Center_x-Width/2.0)+0.5), int((Center_x+Width/2.0)+0.5)])

    if (bounds[0]< 0):
        bounds[1] += -bounds[0]
        bounds[0] = 0
    elif (bounds[1]>=Size_of_Image_y):
        bounds[0] += bounds[1]-(Size_of_Image_y+1)
        bounds[1] = Size_of_Image_y-1

    
    if (bounds[2]< 0):
        bounds[3] += -bounds[2]
        bounds[2] = 0
    elif (bounds[3]>=Size_of_Image_x):
        bounds[2] += bounds[3]-(Size_of_Image_x+1)
        bounds[3] = Size_of_Image_x-1

    if(Height > Width):
        move = int(0.5*(Height-Width)+0.5)
        tmp_img[0:bounds[1]-bounds[0], move:bounds[3]-bounds[2]+move,:] = Received_Im[bounds[0]:bounds[1],bounds[2]:bounds[3],:]
    elif (Height < Width):
        move = int(0.5*(Width-Height)+0.5)
        tmp_img[move:bounds[1]-bounds[0]+move, 0:bounds[3]-bounds[2],:] = Received_Im[bounds[0]:bounds[1],bounds[2]:bounds[3],:]
    else:
        tmp_img[0:bounds[1]-bounds[0], 0:bounds[3]-bounds[2],:] = Received_Im[bounds[0]:bounds[1],bounds[2]:bounds[3],:]

    return cv2.resize(tmp_img,(128,128),interpolation=cv2.INTER_CUBIC)


def AAE(Index, q_BB,q_im,q_out,Internal_Path,Workspace_Path,Show):

    full_name = Internal_Path.split('/')
    workspace_path = Workspace_Path
    Queue_BB = q_BB
    Queue_im = q_im
    Queue_out = q_out
    print workspace_path

    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    os.environ['AE_WORKSPACE_PATH'] = workspace_path

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
    ckpt_dir = utils.get_checkpoint_dir(log_dir)

    codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
    sess = tf.Session()
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    Running = True
    print "Multiprocess setup done " + str(Index)+ " "+os.environ.get('AE_WORKSPACE_PATH')
    time.sleep(3)
    while Running:
        if not(Queue_im.empty() == True):

            Received_Im = Queue_im.get()

            while not(Queue_BB.empty() == True):
                tmp = Queue_BB.get()  
   
                crop_image = get_image_for_AAE(Index,tmp,Received_Im)
                
                R = codebook.nearest_rotation(sess, crop_image)
                
                if(Show):
                    pred_view = dataset.render_rot(R,downSample = 1)

                    cv2.imwrite('AAE_real'+str(Index)+'.png',Received_Im)
                    cv2.imwrite('AAE_prediction'+str(Index)+'.png',pred_view)
                    cv2.imwrite('AAE_crop'+str(Index)+'.png',crop_image)
                                    
                Queue_out.put([R,tmp[5], tmp[6]])
                Queue_BB.task_done()

def rot_mat_to_euler(r): #https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
    if (r[0, 2] == 1) | (r[0, 2] == -1): # special case 
        e3 = 0 # set arbitrarily 
        dlt = np.arctan2(r[0, 1], r[0, 2]) 
        if r[0, 2] == -1: 
            e2 = np.pi/2 
            e1 = e3 + dlt 
        else: 
            e2 = -np.pi/2 
            e1 = -e3 + dlt 
    else: 
        e2 = -np.arcsin(r[0, 2]) 
        e1 = np.arctan2(r[1, 2]/np.cos(e2), r[2, 2]/np.cos(e2))
        e3 = np.arctan2(r[0, 1]/np.cos(e2), r[0, 0]/np.cos(e2)) 
    return e1, e2, e3

if __name__ == '__main__':

    print("Setup initialised")

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
    argv = 9 #number of elements in argc:
    # test files: .data .cfg .weight -dont_show -thresh 0.5
    # change 0.5 to change thresh 
    argc = (c_char_p*argv)('./darknet', 'detector','runYolov3','Batchelor.data', 'CAD_MODEL.cfg', 'yoloweights/Real_model_best.weights', '-dont_show', '-thresh', '0.5') 
    
    
    Yolo_setup(argv,argc) #Setup of Yolo
    Yolo_loadObj() #load figure parameters for depth estimation

    #Setup of AAE:
    Queue_im = []
    Queue_BB = []
    Queue_out = []
    Internal_Path = []
    Workspace_Path = []

    AAE_Process = []

    Number_of_AAE = 3
    Show = 0

    for i in range(Number_of_AAE):
        temp_BB = mp.JoinableQueue()
        Queue_BB.append(temp_BB)
        temp_im = mp.JoinableQueue()
        Queue_im.append(temp_im)
        temp_out = mp.JoinableQueue()
        Queue_out.append(temp_out)
    
    Internal_Path.append("exp_group/my_autoencoder")
    Internal_Path.append("exp_group/my_autoencoder")
    Internal_Path.append("exp_group/my_autoencoder")

    Workspace_Path.append("/home/peter/Documents/darknet2/darknet/AugmentedAutoencoder/AAE_W_BALLJOINT/")
    Workspace_Path.append("/home/peter/Documents/darknet2/darknet/AugmentedAutoencoder/AAE_W/")
    Workspace_Path.append("/home/peter/Documents/darknet2/darknet/AugmentedAutoencoder/AAE_W_PIKACHU/")
    Running_AAE = []
    for i in range(Number_of_AAE):
        temp_process = mp.Process(target=AAE, args=(i, Queue_BB[i] ,Queue_im[i],Queue_out[i],Internal_Path[i],Workspace_Path[i],Show))
        AAE_Process.append( temp_process )
        
        AAE_Process[i].start()
        Running_AAE.append(0)
        time.sleep(3)
        print("Starting the multiprocesses " + str(i))

    #camera setup
    Livefootage = 0 # set to 0 if video is used
    video_path = 'Depth_fil_Test2.bag'
    #camera = cam.Realsense_Camera(640,480,video_path,Livefootage)

    print("Setup complete..")

    pressed = 0
    time_file = open("timings.txt",'a')
    fps_file = open("timings_fps.txt",'a')  
    
    numImg = -1

    for test in range(0, 10):
        numImg = -1
        t0 = time.time()
        camera = cam.Realsense_Camera(640,480,video_path,Livefootage)
        while cam.Realsense_Check_Video_ended(camera): 

            if pressed == 116: # 116 = t
                break

            img = cam.Realsense_Frame_Color(camera)
            
            if len(img) < 3:
                continue

            #MAT translation to char array
            img_res = cv2.resize(img,(608,608))
            frameArray1_tmp = np.fromstring(img_res,np.uint8)
            frameArray2_tmp = np.reshape(frameArray1_tmp,(608,608,3))
            framearray = frameArray2_tmp.tostring()
            
            img_width, img_height, _ = img.shape
        
            numImg += 1 # image number! needs to be incremented for each frame
            t0_yolo = time.time()
            numDet = Yolo_run(framearray) #run yolo return number of detections
            t1_yolo = time.time() - t0_yolo
            if numDet > 0:
                Yolo_getbb.restype = ndpointer(dtype=ctypes.c_float, shape=(numDet,7)) #allocate space in memory for detection matrix

                det_matrix = Yolo_getbb(numDet, numImg, 608, img_width, img_height) #returns detection matrix using format:
                #class, x, y, width, height, image number, Detection element
                
                pose_tmp = []
                pose = [] 

                for rows in det_matrix:
                    if rows[0] != -1:
                        pose_tmp.append(rows)
                        
                        Queue_BB[int(rows[0])].put(rows)
                        Running_AAE[int(rows[0])] = 1

                        while Queue_BB[int(rows[0])].empty(): #wait for queues to be ready
                            pass
                t0_aae = time.time()
                for i in range(Number_of_AAE):
                    if Running_AAE[i]:
                        Queue_im[i].put(img)
                        
                pose = get_Translation2d(pose_tmp) # get distance to object (x, y)
                        
                for i in range(len(pose_tmp)):
                    pose_tmp2 = (ctypes.c_float*len(pose_tmp[i]))(*pose_tmp[i])
                    pose[i].append(Yolo_getDepth(pose_tmp2,img_width, img_height)) #estimate depth

                #join processes
                for i in range(Number_of_AAE):
                    if Running_AAE[i]:

                        Queue_BB[i].join()
                t1_aae = time.time() - t0_aae        
                #insert combination of pose and rotation

                for rows in pose:
                    tmp = Queue_out[int(rows[2])].get()
            
                    if rows[0] == tmp[1] and rows[1] == tmp[2]:
                        thetax, thetay, thetaz = rot_mat_to_euler(tmp[0])
                        rows.append(thetax)
                        rows.append(thetay)
                        rows.append(thetaz)
                    else:
                        print("Error m8") #terminate
                time_file.write(str(test) + "," + str(t1_yolo) + "," + str(t1_aae) + "\n")
                Running_AAE = np.zeros(Number_of_AAE)
                #print(pose) # imageNr, detectionNr, class, x, y, z, thetaX, thetaY, thetaZ
                #x and y is distance from image center

            #cv2.imshow("show image",img)
            #pressed = cv2.waitKey(1)
        t1 = time.time()-t0
        fps_file.write(str(test) + "," + str(numImg/t1) + "," + str(numImg) + "," + str(t1) + "\n")
        print "FPS: " + str(numImg/t1)

    #Free memory
    for i in range(Number_of_AAE):
        AAE_Process[i].terminate()
    fps_file.close()
    time_file.close()
    Yolo_end() 
    cam.Realsense_Terminate(camera)
