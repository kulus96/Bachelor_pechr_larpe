
import numpy.ctypeslib as ctl
import ctypes
import cv2
import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer

def get_Translation2d(arr):
    pose = []
    im_width = 640
    im_height = 480
    mid_x = im_width/2
    mid_y = im_height/2

    for row in range(len(arr)):
        
        pose.append([arr[row][0],abs(arr[row][1]*im_width-mid_x), abs(arr[row][2]*im_height-mid_y)]) #class
        print(pose)

    return pose


if __name__ == '__main__':
    img = cv2.imread('Images/11.jpg',255)
    img2 = cv2.resize(img,(640,480))
    res_img = cv2.resize(img,(608,608))
    #cv2.imshow("Test", img)

    testarray1 = np.fromstring(res_img,np.uint8)
    testarray2 = np.reshape(testarray1,(608,608,3))
    framearray = testarray2.tostring()

    #set library
    libname = 'libdarknet.so'
    libdir = './'
    lib=ctl.load_library(libname,libdir)

    #set method to be called
    py_test = lib.test_py
    py_test2 = lib.test_py2
    py_Det = lib.run_detector
    py_run = lib.runDetectorPy
    py_getbb = lib.get_bb
    py_end = lib.endDetectorPy

    #argument types
    py_test.argtypes = [ctypes.c_int]
    py_test2.argtypes = [ctypes.c_int, ctypes.c_char_p]

    py_Det.argtypes = [ctypes.c_int, POINTER(c_char_p)]
    py_getbb.argtypes = [ctypes.c_int]    
    py_run.argtypes = [ctypes.c_char_p]
    #set return types

    py_test.restype = ndpointer(dtype=ctypes.c_float, shape=(2,5))
    py_run.restype = ctypes.c_int



    #set arg values

    argv = 6
    argc = (c_char_p*argv)('./darknet', 'detector','runYolov3','Batchelor.data', 'CAD_MODEL.cfg', 'yoloweights/Real_model_best.weights') #5 test files: .data .cfg .weight img

    py_Det(argv,argc)
    numDet = py_run(framearray)
    py_getbb.restype = ndpointer(dtype=ctypes.c_float, shape=(numDet,5))
    res = py_getbb(numDet)
    print(res)
    
    detMat = []
    for i in res:
        if i[0] != -1:
            detMat.append(i)

    pose = get_Translation2d(detMat)    
    #print(pose)
    
    py_end()

   
