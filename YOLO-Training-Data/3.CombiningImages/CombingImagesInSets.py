"""
Taking real images and make copies of them with noise.
After divide all the images into sets
Author:lrpd/larpe17
02-03-2019
"""

import cv2
import random
from PIL import Image
import numpy as np
import os
import time
import progressbar
import shutil
import copy

Input_Path = "Images/"
Paths_To_Images = []
Paths_To_txt = []
Output_Path = "Data/"
Unused_Image_indexs = []
Paths_To_Sets = ["Train/","Validation/","Test/"]
Set_Name =["Train","Validation","Test"]

Size_Of_Training_Set = 50
Size_Of_Test_Set = 0
Size_Of_Validation_Set = 50

random.seed(a=None)

start = time.time()  

if (os.path.exists(Output_Path)):
    shutil.rmtree(Output_Path)
os.mkdir(Output_Path)
os.mkdir(Output_Path+Paths_To_Sets[0])
os.mkdir(Output_Path+Paths_To_Sets[1])
os.mkdir(Output_Path+Paths_To_Sets[2])

# ----------- retrieve all files ------------
Images_And_txt =  os.listdir(Input_Path)
for file in Images_And_txt:
    if(not( file.find(".txt")==-1 )):
        Paths_To_txt.append(Input_Path+file)
    else:
        Paths_To_Images.append(Input_Path+file)
Paths_To_Images.sort()
Paths_To_txt.sort()


if(not(np.shape(Paths_To_Images) == np.shape(Paths_To_txt))):
    print("Not the same number of .txt files as image files")
    print("Number of images " + str(np.shape(Paths_To_Images)))
    print("Number of txt " + str(np.shape(Paths_To_txt)))
    print("Program terminates")
    exit()

for i in range(np.shape(Paths_To_Images)[0]):
    if not(Paths_To_Images[i][:-4] == Paths_To_txt[i][:-4]):
            print("Can not find a image and a txt that matches")
            print("Program terminates")
            exit()
# ----------- Dividing the images and annotation in to sub groups -------------

Unused_Image_Indexs = np.arange(np.shape(Paths_To_Images)[0])
Set_Index = 0

Number_Of_images_In_Set = [len(Paths_To_Images)*(Size_Of_Training_Set*0.01),len(Paths_To_Images)*(+Size_Of_Validation_Set*0.01),len(Paths_To_Images)*(Size_Of_Test_Set*0.01)]

print("Dividing the images into groups random")
Number_of_images = 0
for progress in progressbar.progressbar(range(len(Paths_To_Images))):
    Number_of_images = Number_of_images + 1
    
    if Number_Of_images_In_Set[Set_Index] <= Number_of_images and not (Set_Index == 2):
        Set_Index = Set_Index + 1
        Number_of_images = 0
        if Number_Of_images_In_Set[Set_Index] == 0 and not (Set_Index == 2):
            Set_Index = Set_Index + 1
    elif Number_Of_images_In_Set[Set_Index] == 0 and (Set_Index == 2):
        break

    # ----------- Find the image ------------
    Unused_Index = int(random.uniform(0,len(Unused_Image_Indexs)))
    Image_Index = Unused_Image_Indexs[Unused_Index]
    Unused_Image_Indexs = np.delete(Unused_Image_Indexs,Unused_Index)
    Current_Image = cv2.imread(Paths_To_Images[Image_Index])
    cv2.imwrite(Output_Path+Paths_To_Sets[Set_Index]+Paths_To_Images[Image_Index][len(Input_Path):],Current_Image)

    with open(Output_Path+Paths_To_Sets[Set_Index]+ Paths_To_txt[Image_Index][len(Input_Path):], 'a') as file_1, open(Paths_To_txt[Image_Index], 'r') as file_2:
        for line in file_2:
            file_1.write(line)
        file_2.close

    # ------------ Create set file ---------------------#
    Set_file = open(Output_Path + Set_Name[Set_Index] + ".txt","a+")
    Set_file.write( Output_Path + Set_Name[Set_Index] + "/" + Paths_To_Images[Image_Index][len(Input_Path):] + "\n")
    Set_file.close

print ("Dividing the images into sets " + str(time.time()-start) + " seconds.")
