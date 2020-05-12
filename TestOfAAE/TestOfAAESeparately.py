import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser
import random
import math
import shutil
import yaml
import threading


from auto_pose.ae import factory
from auto_pose.ae import utils

## TO RUN IT:
# export AE_WORKSPACE_PATH=/home/lpe/AugmentedAutoencoder/AAE_W/
# python /home/lpe/Desktop/Bachelor/Gen_Data_AAE.py exp_group/my_autoencoder

Number_of_Ran_Images = 200

Output_Path = "/home/lpe/Desktop/AAE_Test/Output/"
Path_to_Background = "/home/lpe/Desktop/AAE_Test/Background/"
Name_of_Object = "Cube"
Size_Of_out_Image = 608
Paths_To_Background_Images = []
Random_Rot = []
Matrix_Data = []
Angles_Data = []
Precision_Data = []

if not os.path.exists(Output_Path):
    os.mkdir(Output_Path)

if (os.path.exists(Output_Path+Name_of_Object+"/")):
    shutil.rmtree(Output_Path+Name_of_Object+"/")
os.mkdir(Output_Path+Name_of_Object+"/")
os.mkdir(Output_Path+Name_of_Object+"/ROT/")
os.mkdir(Output_Path+Name_of_Object+"/Angles/")
os.mkdir(Output_Path+Name_of_Object+"/Images/")


def Ran_Image(alpha,beta,gamma):
    R[0][0]= math.cos(alpha)*math.cos(beta)
    R[0][1]= math.cos(alpha)*math.sin(beta)*math.sin(gamma)-math.sin(alpha)*math.cos(gamma) #Rotation matrix
    R[0][2]= math.cos(alpha)*math.sin(beta)*math.cos(gamma)+math.sin(alpha)*math.sin(gamma)
    R[1][0]= math.sin(alpha)*math.cos(beta)
    R[1][1]= math.sin(alpha)*math.sin(beta)*math.sin(gamma)+math.cos(alpha)*math.cos(gamma)
    R[1][2]= math.sin(alpha)*math.sin(beta)*math.cos(gamma)-math.cos(alpha)*math.sin(gamma)
    R[2][0]= -math.sin(beta)
    R[2][1]= math.cos(beta)*math.sin(gamma)
    R[2][2]= math.cos(beta)*math.cos(gamma)
    return R , dataset.render_rot(R,downSample = 1)

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

def rot_mat_to_euler2(r): 
    if not((r[2,0] == 1) | (r[0, 2] == -1)):
        e21 = - np.arcsin(r[2,0])
        e22 = math.pi-e21
        e11 = np.arctan2(r[2,1]/np.cos(e21), r[2,2]/np.cos(e21))
        e12 = np.arctan2(r[2,1]/np.cos(e22), r[2,2]/np.cos(e22))

        e31 = np.arctan2(r[1,0]/np.cos(e21), r[0,0]/np.cos(e21))
        e32 = np.arctan2(r[1,0]/np.cos(e22), r[0,0]/np.cos(e22))

    else:
        e31 = 0
        if r[2,0] == -1:
            e21 = math.pi/2
            e11 = e31 + np.arctan2(r[0,1],r[0,2])
        else:
            e21 = -math.pi/2
            e11 = -e31 + np.arctan2(-r[0,1],-r[0,2])
    return e11, e21, e32

def rot_mat_to_euler3(r): #https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
    e1 = math.atan2(r[2,1],r[2,2])
    e2 = math.atan2(-r[2,0],math.sqrt(r[2,1]*r[2,1]+r[2,2]*r[2,2]))
    e3 = math.atan2(r[1,0],r[0,0])
    return e1, e2, e3



bg = os.listdir(Path_to_Background)
for file in bg:
    Paths_To_Background_Images.append(Path_to_Background+file)
    



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
arguments = parser.parse_args()

full_name = arguments.experiment_name.split('/')

experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

print(experiment_name)
print(experiment_group)

codebook,dataset = factory.build_codebook_from_name(experiment_name,experiment_group,return_dataset=True)
R = np.eye(3)
alpha = 0
beta = 0
gamma = 0



workspace_path = os.environ.get('AE_WORKSPACE_PATH')
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True


Scale_Test = [20,40,60,80,100,120,140,160,180,200]

Light_Test = [10,20,40,60,80,100,120,140,160,180,200]

Noise_Test = [0,10,20,30,40,50,60,70,80,90,100]

Number_of_Scale = 10
Number_of_Light = 11
Number_of_Noise = 10
Number_of_Blur = 10

Test_Number = 0

Total_Number_of_test = 127 #Number_of_Color*Number_of_Light*Number_of_Noise*Number_of_Scale

Mid_Scale = 4
Mid_Light = 5
Mid_Noise = 0
Mid_Blur = 0
 
Total_Precision_accepted = 0
Total_Precision_declined = 0


run = 0
with tf.Session(config=config) as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)
    for scale in range(Number_of_Scale):
        for blur in range(Number_of_Blur):
            for light in range(Number_of_Light):
                for noise in range(Number_of_Noise):
                    Test_vali = 0
                    if not scale == Mid_Scale:
                        Test_vali = Test_vali + 1 
                    if not blur == Mid_Blur:
                        Test_vali = Test_vali + 1
                    if not light == Mid_Light:
                        Test_vali = Test_vali + 1
                    if not noise == Mid_Noise:
                        Test_vali = Test_vali + 1

                    #Test_Number = Test_Number + 1

                    if Test_vali <= 1 :
                        Matrix_Data = []
                        Angles_Data = []
                        
                        Test_Number = Test_Number + 1
                        if Test_Number == 1 or run == 1:
                            run = 1
                            Data_file_ROT = open(Output_Path + Name_of_Object + "/" + "ROT/" + str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + ".csv","a+")
                            Data_file_ROT.seek(0)
                            Data_file_Angles = open(Output_Path + Name_of_Object + "/" + "Angles/" + str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + ".csv","a+")
                            Data_file_Angles.seek(0)

                            Precision_accepted = 0
                            Precision_declined = 0


                            for progress in range(Number_of_Ran_Images):
                                print("Test Number: " + str(Test_Number) + " out of: "+ str(Total_Number_of_test) + " Scale: " + str(Scale_Test[scale]) + " Blur: " + str(blur) +" Light: " + str(Light_Test[light]) + " Noise: " + str(Noise_Test[noise]) + " Progress: " + str(progress))
                                
                                alpha = random.random()*2*math.pi-math.pi
                                beta = random.random()*2*math.pi-math.pi
                                gamma = random.random()*2*math.pi-math.pi
                                Ran_Rot_Object = []
                                Ran_Rot , Ran_Rot_Object = Ran_Image(alpha,beta,gamma)

                                out_image = cv2.imread(Paths_To_Background_Images[int(random.uniform(0,len(Paths_To_Background_Images)-1))])
                                out_image = cv2.resize(out_image,(Size_Of_out_Image,Size_Of_out_Image),interpolation=cv2.INTER_CUBIC)

                                # ----------- Resizing the image ------------

                                (Size_Of_Input_image_y,Size_Of_Input_image_x,Output_Channels) = Ran_Rot_Object.shape
                                Scale_value = Scale_Test[scale]*0.01
                                Resized_Ran_Rot_Object = cv2.resize(Ran_Rot_Object,(int(Size_Of_Input_image_y*Scale_value),int(Size_Of_Input_image_x*Scale_value)))

                                # ----------- Finding the contour of the object in the image ------------
                                gray_image = cv2.cvtColor(Resized_Ran_Rot_Object, cv2.COLOR_BGR2GRAY)
                                ret, thresh = cv2.threshold(gray_image, 15, 15,cv2.THRESH_BINARY)
                                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                                mask = np.zeros_like(Resized_Ran_Rot_Object) # Create mask where white is what we want, blackotherwise
                                cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask

                                (y, x, BGR) = np.where(mask == 255)
                                #cv2.imshow("mask",mask)
                                #cv2.waitKey(0)
                                if len(x) == 0 or len(y) == 0:
                                    print(Ran_Rot)
                                    print(alpha,beta,gamma)
                                    cv2.imwrite(Output_Path + "/" + str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_" + str(progress) + "_object"  + ".jpg",Ran_Rot_Object)
                                    cv2.imwrite(Output_Path + "/" + str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_" + str(progress) + "_mask"  + ".jpg",mask)

                                (bottomy, bottomx) = (np.min(y), np.min(x))
                                (topy, topx) = (np.max(y), np.max(x))

                                # ----------- Crop the image and copying it into the background ------------
                                offset = 608/2
                                for l in range(len(x)):
                                    if not(np.any(Resized_Ran_Rot_Object[y[l],x[l]]) == 0):
                                        out_image[y[l]+offset,x[l]+offset]=Resized_Ran_Rot_Object[y[l],x[l]]                    

                                # ----------- Add random color filtering, brightness and contrast ------------
                                
                                # brightness and saturation
                                

                                hls = cv2.cvtColor(out_image, cv2.COLOR_BGR2HLS)
                                h, l, s = cv2.split(hls)
    
                                Lightness_New = cv2.multiply(l,Light_Test[light]*0.01)

                                Lightness_New[Lightness_New > 255] = 255

                                Final_hls = cv2.merge((h, Lightness_New, s))
                                out_image= cv2.cvtColor(Final_hls, cv2.COLOR_HLS2BGR)

                                # Adding Gaussian Blur
                                Kernel_size= 2*blur-1#random.randint(0,6)
                                if not(blur==0):
                                    out_image = cv2.GaussianBlur(out_image,(Kernel_size,Kernel_size),cv2.BORDER_DEFAULT) 

                                # -------- Adding Gaussian noise ------

                                if not (noise == 0):
                                
                                    sigma = ((256)/2)*Noise_Test[noise]*0.01
                                    mu = 0
                                    out_image = np.asarray(out_image, dtype=int)

                                    Gaussian_Noise_temp = np.random.normal(mu, sigma,(out_image.shape[0],out_image.shape[1],3))
                                    Gaussian_Noise_temp = Gaussian_Noise_temp.astype(int)

                                    out_image += Gaussian_Noise_temp

                                    out_image[:,:,0][out_image[:,:,0]>255] = 255
                                    out_image[:,:,1][out_image[:,:,1]>255] = 255
                                    out_image[:,:,2][out_image[:,:,2]>255] = 255
                                    out_image[:,:,0][out_image[:,:,0]<0] = 0
                                    out_image[:,:,1][out_image[:,:,1]<0] = 0
                                    out_image[:,:,2][out_image[:,:,2]<0] = 0

                                    out_image = np.asarray(out_image, dtype=np.uint8)
                                    
                                    

                                # -------- Cropping the image as YOLO would ------
                                Width = float(topx-bottomx)
                                Height = float(topy-bottomy)
                                Center_y = float(Height/2+bottomy)
                                Center_x = float(Width/2+bottomx)

                                if (Width < Height):
                                    add_Width = (Height-Width)/2
                                    add_Height = 0    
                                else:
                                    add_Height = (Width-Height)/2
                                    add_Width = 0  
                                
                                
                                crop_image = out_image[int(Center_y-Height/2-add_Height+offset):int(Center_y+Height/2+add_Height+offset), int(Center_x-Width/2-add_Width+offset):int(Center_x+Width/2+add_Width+offset)]

                                crop_image = cv2.resize(crop_image,(128,128),interpolation=cv2.INTER_CUBIC)
                                
                                # ------------ Save image ---------------------
                                if progress < 3 : 
                                    cv2.imwrite(Output_Path + Name_of_Object + "/" + "Images/"+str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_" + str(progress) + "_crop"  + ".jpg",crop_image)
                                    cv2.imwrite(Output_Path + Name_of_Object + "/" + "Images/"+str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_" + str(progress) + "_out"  + ".jpg",out_image)
                                    cv2.imwrite(Output_Path + Name_of_Object + "/" + "Images/"+str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_" + str(progress) + "_object"  + ".jpg",Ran_Rot_Object)
                                
                                """
                                    cv2.imshow(str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_1", crop_image)
                                    cv2.imshow(str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_2", out_image)
                                    cv2.imshow(str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "_3", Resized_Ran_Rot_Object)
                                """
                                #cv2.waitKey(0)
                                
                                R_predict = codebook.nearest_rotation(sess, crop_image)

                                R_predict_inv = np.linalg.inv(R_predict)

                                temp = np.dot(Ran_Rot,R_predict_inv)
                                
                                a , b , g = rot_mat_to_euler(temp)
                                """
                                orig = temp
                                print(temp-orig)
                                Ran_Rot_temp , Ran_Rot_Object_temp = Ran_Image(a,b,g)
                                print(1)
                                print(Ran_Rot_temp-orig)
                                a , b , g = rot_mat_to_euler2(temp)
                                Ran_Rot_temp , Ran_Rot_Object_temp = Ran_Image(a,b,g)
                                print(2)
                                print(Ran_Rot_temp-orig)
                                a , b , g = rot_mat_to_euler3(temp)
                                Ran_Rot_temp , Ran_Rot_Object_temp = Ran_Image(a,b,g)
                                print(3)
                                print(Ran_Rot_temp-orig)
                                """

                                #print(alpha,beta,gamma)
                                #print(rot_mat_to_euler(temp))
                                #print rot_mat_to_euler2(temp)
                                cv2.waitKey(0)

                                Angles_Data.append([a,b,g])

                                Matrix_Data.append(sum(sum(abs(temp-np.eye(3))))/9)
                                # ------------ save data file ---------------------#
                                
                                Data_file_ROT.write(str(progress) + "," + str(Ran_Rot[0][0]) + "," + str(Ran_Rot[0][1]) + "," +str(Ran_Rot[0][2]) + "," +str(Ran_Rot[1][0]) + "," +str(Ran_Rot[1][1]) + "," +str(Ran_Rot[1][2]) + "," +str(Ran_Rot[2][0]) + "," +str(Ran_Rot[2][1]) + "," +str(Ran_Rot[2][2]) + "," + str(R_predict[0][0]) + "," + str(R_predict[0][1]) + "," +str(R_predict[0][2]) + "," + str(R_predict[1][0]) + "," + str(R_predict[1][1]) + "," + str(R_predict[1][2]) + "," + str(R_predict[2][0]) + "," + str(R_predict[2][1]) + "," + str(R_predict[2][2]) + "\n")
                                Data_file_Angles.write(str(progress) + "," + str(a) + "," + str(b) + "," + str(g) + "\n")

                                
                                # ------------ Precision data ---------------------#
                                Threshold_precision = math.pi*0.025
                                if(abs(a) > Threshold_precision) or  (abs(b) > Threshold_precision) or  (abs(g) > Threshold_precision):
                                    Precision_declined = Precision_declined + 1
                                else:
                                    Precision_accepted = Precision_accepted +1 



                            
                            Data_file_ROT.close
                            Data_file_Angles.close
                            
                            Data_file_precision = open(Output_Path + Name_of_Object + "/" + "precision_" + Name_of_Object  + ".txt","a+")
                            Data_file_precision.seek(0)
                            Data_file_precision.write(str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "\n")
                            Data_file_precision.write(str(Precision_accepted) + "," + str(Precision_declined) + "\n")
                            Data_file_precision.close

                            Total_Precision_accepted += Precision_accepted
                            Total_Precision_declined += Precision_declined

                            Angles_avg_abs = np.sum(np.absolute(Angles_Data),axis=0)/len(Angles_Data)
                            Data_file_Angles = open(Output_Path + Name_of_Object + "/" + "Angle_Data_abs_" + Name_of_Object +  ".txt","a+")
                            Data_file_Angles.seek(0)
                            Data_file_Angles.write(str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "\n")
                            Data_file_Angles.write( str(Angles_avg_abs[0]) + "_" + str(Angles_avg_abs[1]) + "_" + str(Angles_avg_abs[2]) + "\n")
                            Data_file_Angles.close
                            
                            Data_file_ROT = open(Output_Path + Name_of_Object + "/" + "ROT_Data_" + Name_of_Object  + ".txt","a+")
                            Data_file_ROT.seek(0)
                            Data_file_ROT.write(str(scale) + "_" + str(blur) + "_" + str(light) + "_" + str(noise) + "\n")
                            Data_file_ROT.write( str(sum(Matrix_Data)/len(Matrix_Data))+ " " + str(max(Matrix_Data)) + " " + str(min(Matrix_Data)) + "\n")
                            Data_file_ROT.close
                            print("Avg: " + str(sum(Matrix_Data)/len(Matrix_Data)) + " Max: " + str(max(Matrix_Data)) + " Min: " + str(min(Matrix_Data)))
                            print("Precision_accepted: " + str(Precision_accepted) + " Precision_declined: " + str(Precision_declined))
                            print("Abs Avg: " + str(Angles_avg_abs))
print("Total_Precision_accepted: " + str(Total_Precision_accepted) + " Total_Precision_declined: " + str(Total_Precision_declined))

