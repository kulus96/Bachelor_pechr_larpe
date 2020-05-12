import cv2
import imageio
import numpy as np
import pyrealsense2 as rs
import os
import shutil


class Realsense_Camera:
    def __init__(self, Image_Width,Image_Height,File_Path, Bool_live):
        self.Image_Height = Image_Height
        self.Image_Width = Image_Width
        self.pipeline = rs.pipeline()
        self.File_Path = File_Path
        self.Bool_live = Bool_live
        self.depth_image = np.zeros(1)
        self.color_image = np.zeros(1)
        self.ready = 0
        if(self.Bool_live):
                Realsense_Setup_live(self)
        else:
                Realsense_Setup_recorded(self)
	    

def Realsense_Setup_live(self):
    self.config = rs.config()
    self.config.enable_stream(rs.stream.color, self.Image_Width, self.Image_Height, rs.format.bgr8, 30)
    self.config.enable_stream(rs.stream.depth, self.Image_Width, self.Image_Height, rs.format.z16, 30)
    self.profile = self.pipeline.start(self.config)
    self.align_to = rs.stream.color
    self.align = rs.align(self.align_to)

    self.depth_sensor = self.profile.get_device().first_depth_sensor()
    self.depth_sensor.set_option(rs.option.visual_preset, 3)  # Set high accuracy for depth sensor
    self.depth_scale = self.depth_sensor.get_depth_scale()

def Realsense_Setup_recorded(self):
    self.config = rs.config()
    self.config.enable_stream(rs.stream.color, self.Image_Width, self.Image_Height, rs.format.bgr8, 30)
    self.config.enable_stream(rs.stream.depth, self.Image_Width, self.Image_Height, rs.format.z16, 30)
    
    self.config.enable_device_from_file(self.File_Path,repeat_playback=False)
    self.profile = self.pipeline.start(self.config)
    self.align_to = rs.stream.color
    self.align = rs.align(self.align_to)

    self.depth_sensor = self.profile.get_device().first_depth_sensor()
    self.depth_scale = self.depth_sensor.get_depth_scale()
    self.playback = rs.playback(self.profile.get_device())



def Realsense_Frame_Color(self):
    
    self.frames = self.pipeline.wait_for_frames()
    
    self.aligned_frames = self.align.process(self.frames)
    
    color_frame = self.aligned_frames.get_color_frame()
    if not color_frame:
	    self.color_image = np.zeros(1)
    else:
	    # Convert images to numpy arrays
	    self.color_image = np.asanyarray(color_frame.get_data())
    return self.color_image

def Realsense_Frame_Depth(self):
    depth_frame = self.aligned_frames.get_depth_frame()
    if not depth_frame:
	    self.depth_image = np.zeros(1)
    else:
	    # Convert images to numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
    return self.depth_image

def Realsense_get_Distance(self,x,y,area,show):
    dist,_,_,_ = cv2.mean(self.depth_image[y-area:y+area,x-area:x+area] * self.depth_scale)
    cv2.imshow("test2",self.color_image[y-area:y+area,x-area:x+area])
    if(show):
        cv2.imshow("Depth image",cv2.rectangle(self.Depth_color_Map, (x-area,y-area), (x+area,y+area), (0, 0, 255), 2) )
    return dist

def Realsense_Show_Depth(self):
    self.Depth_color_Map=cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow("Depth image",self.Depth_color_Map)

def Realsense_Terminate(self):
    self.pipeline.stop()

def Realsense_Check_Video_ended(self):
    if not (self.Bool_live):
        if self.ready == 1:
            return self.playback.current_status() == rs.playback_status.playing
        else:
            if (self.playback.current_status() == rs.playback_status.playing):
                self.ready = 1
            return 1
    else:
        return 1