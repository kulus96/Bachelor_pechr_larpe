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
Paths_To_Real_Images = []
Paths_To_Real_txt = []
Temp_Path = "Temp/"
Output_Path = "Data/"
Unused_Image_indexs = []
Paths_To_Sets = ["Train/","Validation/","Test/"]
Set_Name =["Train","Validation","Test"]

Number_of_fake_Images = 9  #Set this to determind how many duplicates of a single image is wanted

Size_Of_Training_Set = 60
Size_Of_Test_Set = 20
Size_Of_Validation_Set = 20



random.seed(a=None)

start = time.time()  

if (os.path.exists(Output_Path)):
    shutil.rmtree(Output_Path)
os.mkdir(Output_Path)
os.mkdir(Output_Path+Paths_To_Sets[0])
os.mkdir(Output_Path+Paths_To_Sets[1])
os.mkdir(Output_Path+Paths_To_Sets[2])
if (os.path.exists(Temp_Path)):
    shutil.rmtree(Temp_Path)
os.mkdir(Temp_Path)

# ----------- retrieve all files ------------
Images_And_txt =  os.listdir(Input_Path)
for file in Images_And_txt:
    if(not( file.find(".txt")==-1 )):
        Paths_To_Real_txt.append(Input_Path+file)
    else:
        Paths_To_Real_Images.append(Input_Path+file)
Paths_To_Real_Images.sort()
Paths_To_Real_txt.sort()


if(not(np.shape(Paths_To_Real_Images) == np.shape(Paths_To_Real_txt))):
    print("Not the same number of .txt files as image files")
    print("Program terminates")
    exit()

for i in range(np.shape(Paths_To_Real_Images)[0]):
    if not(Paths_To_Real_Images[i][:-4] == Paths_To_Real_txt[i][:-4]):
            print("Can not find a image and a txt that matches")
            print("Program terminates")
            exit()

Unused_Image_Indexs = np.arange(np.shape(Paths_To_Real_Images)[0])
Last_Image_Index = -1
print("Generating the images")
for progress in progressbar.progressbar(range(len(Paths_To_Real_Images)*Number_of_fake_Images)):

    # ----------- Find the image ------------
    if ( progress % Number_of_fake_Images == 0 or progress == 0):
        Unused_Index = int(random.uniform(0,len(Unused_Image_Indexs)))
        Image_Index = Unused_Image_Indexs[Unused_Index]
        Unused_Image_Indexs = np.delete(Unused_Image_Indexs,Unused_Index)
        Number_Of_Fake_Images_Made = 0

    Name = str(Image_Index)

    if( not ( Image_Index == Last_Image_Index)):
        Last_Image_Index = Image_Index
        
    # ----------- Copy the annotation file ------------

        for i in range(Number_of_fake_Images+1):
            with open(Temp_Path + Name +"_" + str(i) + ".txt", 'a') as file_1, open(Paths_To_Real_txt[Image_Index], 'r') as file_2:
                for line in file_2:
                    file_1.write(line)
                file_1.close
        file_2.close

        # ----------- Copy the image ------------
        Current_Image = cv2.imread(Paths_To_Real_Images[Image_Index])
        cv2.imwrite(Temp_Path + Name + "_" + str(Number_Of_Fake_Images_Made) + ".jpg",Current_Image)
        Number_Of_Fake_Images_Made = Number_Of_Fake_Images_Made + 1


    Copy_Of_Current_Image = copy.copy(Current_Image)

    # ----------- Add random color filtering, brightness and contrast ------------


    # color 
    Random_B_Scale = random.randint(50,100)*0.01
    Random_G_Scale = random.randint(50,100)*0.01
    Random_R_Scale = random.randint(50,100)*0.01

    Copy_Of_Current_Image[:,:,0] = Copy_Of_Current_Image[:,:,0]*Random_B_Scale
    Copy_Of_Current_Image[:,:,1] = Copy_Of_Current_Image[:,:,1]*Random_G_Scale
    Copy_Of_Current_Image[:,:,2] = Copy_Of_Current_Image[:,:,2]*Random_R_Scale

    # brightness and saturation
    Random_Saturation_Scale = random.randint(50,130)*0.01
    Random_Value_Scale = random.randint(50,200)*0.01

    hsv = cv2.cvtColor(Copy_Of_Current_Image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    Saturation_New = cv2.multiply(s,Random_Saturation_Scale)
    Value_New = cv2.multiply(v,Random_Value_Scale)

    Final_hsv = cv2.merge((h, Saturation_New, Value_New))
    Copy_Of_Current_Image= cv2.cvtColor(Final_hsv, cv2.COLOR_HSV2BGR)


# ------------ Save image ---------------------
    cv2.imwrite(Temp_Path + Name + "_" + str(Number_Of_Fake_Images_Made) + ".jpg",Copy_Of_Current_Image)

    Number_Of_Fake_Images_Made = Number_Of_Fake_Images_Made + 1
    
# ------------ Show Image ---------------------#
    #cv2.imshow(Temp_Path + Name + "_" + str(Number_Of_Fake_Images_Made), Copy_Of_Current_Image)
    #cv2.waitKey(00)
    #cv2.destroyAllWindows()

# ----------- Dividing the images and annotation in to sub groups -------------
Paths_To_Real_txt = []
Paths_To_Real_Images = []
Images_And_txt =  os.listdir(Temp_Path)

for file in Images_And_txt:
    if(not( file.find(".txt")==-1 )):
        Paths_To_Real_txt.append(Temp_Path+file)
    else:
        Paths_To_Real_Images.append(Temp_Path+file)

Paths_To_Real_Images.sort()
Paths_To_Real_txt.sort()

Unused_Image_Indexs = np.arange(np.shape(Paths_To_Real_Images)[0])
Set_Index = 0

Number_Of_images_In_Set = [len(Paths_To_Real_Images)*(Size_Of_Training_Set*0.01),len(Paths_To_Real_Images)*(+Size_Of_Validation_Set*0.01),len(Paths_To_Real_Images)*(Size_Of_Test_Set*0.01)]
Number_of_images = 0
print("Dividing the images into sets random")
for progress in progressbar.progressbar(range(len(Paths_To_Real_Images))):
    Number_of_images = Number_of_images + 1
    if Number_Of_images_In_Set[Set_Index] <= Number_of_images and not (Set_Index == 2):
        Set_Index = Set_Index + 1
        Number_of_images = 0
        if Number_Of_images_In_Set[Set_Index] == 0 and not (Set_Index == 2):
            Set_Index = Set_Index + 1

    # ----------- Find the image ------------
    Unused_Index = int(random.uniform(0,len(Unused_Image_Indexs)))
    Image_Index = Unused_Image_Indexs[Unused_Index]
    Unused_Image_Indexs = np.delete(Unused_Image_Indexs,Unused_Index)
    Current_Image = cv2.imread(Paths_To_Real_Images[Image_Index])
    cv2.imwrite(Output_Path+Paths_To_Sets[Set_Index]+Paths_To_Real_Images[Image_Index][len(Temp_Path):],Current_Image)

    with open(Output_Path+Paths_To_Sets[Set_Index]+ Paths_To_Real_txt[Image_Index][len(Temp_Path):], 'a') as file_1, open(Paths_To_Real_txt[Image_Index], 'r') as file_2:
        #file_1.write(Paths_To_Real_txt[Image_Index]+ "\n")
        for line in file_2:
            file_1.write(line)
        file_2.close

    # ------------ Create set file ---------------------#
    Set_file = open(Output_Path + Set_Name[Set_Index] + ".txt","a+")
    Set_file.write( Output_Path + Set_Name[Set_Index] + "/" + Paths_To_Real_Images[Image_Index][len(Temp_Path):] + "\n")
    Set_file.close
    
shutil.rmtree(Temp_Path)
print ("Generating images and dividing the images into sets " + str(time.time()-start) + " seconds.")
