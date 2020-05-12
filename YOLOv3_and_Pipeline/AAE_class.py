import threading
import time 
from Queue import Queue
import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser

from auto_pose.ae import factory, utils

print_lock = threading.Lock()

class AAE(threading.Thread):
    def __init__(self,index, queue_BB,queue_im,queue_out,full_name, workspace_path,show_data, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue_BB = queue_BB
        self.queue_im = queue_im
        self.queue_out = queue_out
        self.daemon = True
        self.receive_messages = args[0]
        self.full_name = full_name.split('/')
        self. workspace_path = workspace_path
        self.index = index
        self.show = show_data
        print workspace_path

        self.experiment_name = self.full_name.pop()
        self.experiment_group = self.full_name.pop() if len(full_name) > 0 else ''
        os.environ['AE_WORKSPACE_PATH'] = self.workspace_path

        if self.workspace_path == None:
            print('Please define a workspace path:\n')
            print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
            exit(-1)
        
        self.log_dir = utils.get_log_dir(self.workspace_path,self.experiment_name,self.experiment_group)
        self.ckpt_dir = utils.get_checkpoint_dir(self.log_dir)

        self.codebook, self.dataset = factory.build_codebook_from_name(self.experiment_name, self.experiment_group, return_dataset=True)

        self.gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
        self.config = tf.ConfigProto(gpu_options=self.gpu_options)
        self.config.gpu_options.allow_growth = True

        self.sess = tf.Session()#config=self.config)

        factory.restore_checkpoint(self.sess, tf.train.Saver(), self.ckpt_dir)

        self.running = True


    def run(self):
        print threading.currentThread().getName(), self.receive_messages

        while self.running:
            #print" thread " + str (self.queue_im.empty())
            if not(self.queue_im.empty() == True):

                self.received_image = self.queue_im.get()

                while not(self.queue_BB.empty() == True):     
                    crop_image = get_image_for_AAE(self,self.queue_BB.get())
                    
                    R = self.codebook.nearest_rotation(self.sess, crop_image)
                    
                    if(self.show):
                        pred_view = self.dataset.render_rot( R,downSample = 1)
                        cv2.imshow('resized img', pred_view)
                        cv2.imshow('test img', crop_image)
                        cv2.waitKey(000)
                        cv2.destroyAllWindows()
                    self.queue_out.put(R)

                    
    def destroy(self):
        print "TIME TO DESTROY!"
        self.running = False
        tf.reset_default_graph()

def get_image_for_AAE(self,BB):
    # Yolo format : <object-class> <x> <y> <width> <height>
    if not (self.index == BB[0]):
        print("Bounding box class does not match with the AAE called ", threading.currentThread().getName() )
        exit(0)
    Size_of_Image_y,Size_of_Image_x,_ = self.received_image.shape

    Center_x = BB[1] * Size_of_Image_x
    Center_y = BB[2] * Size_of_Image_y
    Width = BB[3]*Size_of_Image_x
    Height = BB[4]*Size_of_Image_y

    if (Width < Height):
        add_Width = (Height-Width)/2
        add_Height = 0    
    else:
        add_Height = (Width-Height)/2
        add_Width = 0  
    
    crop_image = self.received_image[int(Center_y-Height/2-add_Height):int(Center_y+Height/2+add_Height), int(Center_x-Width/2-add_Width):int(Center_x+Width/2+add_Width)]

    return  cv2.resize(crop_image,(128,128),interpolation=cv2.INTER_CUBIC)

       
