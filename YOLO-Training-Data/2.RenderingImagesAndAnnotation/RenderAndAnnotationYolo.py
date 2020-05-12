"""
Taking rendered images and adding a background and anotation for YOLOv3
Author:lrpd/larpe17
17-02-2020
"""

import cv2
import random
from PIL import Image
import numpy as np
import os
import time
import progressbar

Paths_To_Rendered_Images = []
Paths_To_Background_Images = []
Name_of_Obj = []
Paths_To_Sets = ["Train/","Validation/","Test/"]
Set_name =["Train","Validation","Test"]

random.seed(a=None)
# ----------- Configuration of program ------------
with open('createData.cfg', "r") as f:
    for line in f: 
    #print(line) 
        if line == "\n":
            continue   
        elif line[0] == '#': # a title in the config file 
            words = line[7:].split(" ")
            title = words[0]
            continue
        
        #print(title)
        if title == 'Name':
            #print(line)
            Name_of_Obj.append(line.rstrip())


        elif title == 'Rendered':
            #print(line)
            Path_To_Images_rendered = line.rstrip()
            for i in range(len(Name_of_Obj)):    
                p = os.listdir(Path_To_Images_rendered+Name_of_Obj[i])
                Paths_To_Rendered_Images.append([])
                for file in p:
                    Paths_To_Rendered_Images[i].append(Path_To_Images_rendered+Name_of_Obj[i]+'/'+file)

        elif title == 'Background':
            Path_To_Images_Background = line.rstrip()
            bg = os.listdir(Path_To_Images_Background)
            for file in bg:
                Paths_To_Background_Images.append(Path_To_Images_Background+file)

        elif title == 'Number':
            #print(line)
            Number_Of_repetition = int(line)

        elif title == 'Training':
            #print(line)
            Output_Path = line.rstrip()
            if not os.path.exists(Output_Path):
                os.mkdir(Output_Path)
        
        elif title == 'Size':
            Size_Of_out_Image = int(line)

        elif title == 'Trainingset':
            Size_Of_Training_Set = int(line)
            
        elif title == 'Validationset':
            Size_Of_Validation_Set = int(line)

        elif title == 'Testset':
            Size_Of_Test_Set = int(line)
# ----------- Check if the set size is 100% ------------            

if not(Size_Of_Training_Set+Size_Of_Validation_Set+Size_Of_Test_Set == 100):
    print("Size of sets does not add up")
    print("Program terminates")
    exit()
# ----------- Makes folders for sets ------------    

if not os.path.exists(Output_Path+Paths_To_Sets[0]):
    os.mkdir(Output_Path+Paths_To_Sets[0])

if not os.path.exists(Output_Path+Paths_To_Sets[1]):
    os.mkdir(Output_Path+Paths_To_Sets[1])    

if not os.path.exists(Output_Path+Paths_To_Sets[2]):
    os.mkdir(Output_Path+Paths_To_Sets[2])

# ----------- Start of program ------------            
start = time.time()
Max_itereation=Number_Of_repetition*len(Name_of_Obj)*np.shape(Paths_To_Rendered_Images)[1]
Number_Of_images = 0
Obj_Index = 0
#Image_Index =0.0
Number_Of_images_In_Set = [Max_itereation*(Size_Of_Training_Set*0.01),Max_itereation*(Size_Of_Validation_Set*0.01),Max_itereation*(Size_Of_Test_Set*0.01)]
Set_Index = 0

Test = 0
Test1 = 0


for progress in progressbar.progressbar(range(Max_itereation)):
    random.seed(a=None)
    Number_Of_images = Number_Of_images + 1

    if Number_Of_images_In_Set[Set_Index] == Number_Of_images and not (Set_Index == 2):
        Set_Index = Set_Index + 1
        Number_Of_images = 0
   

    # ----------- Random object and background ------------
    Obj_Index =int(random.uniform(0,1)+0.5) #)np.shape(Paths_To_Rendered_Images)[0]-1))
    Image_Index = int(random.uniform(0,np.shape(Paths_To_Rendered_Images)[1]-1))
    if Obj_Index == 0:
        Test = Test + 1
    else:
        Test1 = Test1 + 1  

    # ----------- Loading the images ------------
    rend_image = cv2.imread(Paths_To_Rendered_Images[Obj_Index][Image_Index])
    out_image = cv2.imread(Paths_To_Background_Images[int(random.uniform(0,len(Paths_To_Background_Images)-1))])

    # ----------- Resizing the image ------------
    (Size_Of_Input_image_y,Size_Of_Input_image_x,Output_Channels) = rend_image.shape
    Scale = random.uniform(1.2,0.5)
    rend_image = cv2.resize(rend_image,(int(Size_Of_Input_image_y*Scale),int(Size_Of_Input_image_x*Scale)))
    #print (rend_image.shape)
    out_image = cv2.resize(out_image,(Size_Of_out_Image,Size_Of_out_Image),interpolation=cv2.INTER_CUBIC)

    # ----------- Finding the contour of the object in the image ------------
    gray_image = cv2.cvtColor(rend_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 10, 10, 10)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    mask = np.zeros_like(rend_image) # Create mask where white is what we want, blackotherwise
    cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask

    (y, x, BGR) = np.where(mask == 255)


    # ----------- Finding a random possition for the objects in the output image ------------
    (bottomy, bottomx) = (np.min(y), np.min(x))
    (topy, topx) = (np.max(y), np.max(x))

    Object_fits = False
    Offset_x = 0
    Offset_y = 0
    (Size_Of_Output_image_y,Size_Of_Output_image_x,Output_Channels) = out_image.shape

    while(not Object_fits):
        Offset_x = int(random.uniform(-Size_Of_Output_image_x/2, Size_Of_Output_image_x/2))
        Offset_y = int(random.uniform(-Size_Of_Output_image_y/2, Size_Of_Output_image_y/2))
        if topx+Offset_x >= 0 and topy+Offset_y >= 0 :
            if bottomx+Offset_x > 0 and bottomy+Offset_y > 0 and topx+Offset_x < Size_Of_Output_image_x and topy+Offset_y < Size_Of_Output_image_y :
                Object_fits = True

    # ----------- Crop the image and copying it into the background ------------
    for l in range(len(x)):
        if not(np.any(rend_image[y[l],x[l]]) == 0):
            out_image[y[l]+Offset_y,x[l]+Offset_x]=rend_image[y[l],x[l]]

        # ----------- Add random color filtering, brightness and contrast ------------
    # color 
    Random_B_Scale = random.randint(50,100)*0.01
    Random_G_Scale = random.randint(50,100)*0.01
    Random_R_Scale = random.randint(50,100)*0.01

    out_image[:,:,0] = out_image[:,:,0]*Random_B_Scale
    out_image[:,:,1] = out_image[:,:,1]*Random_G_Scale
    out_image[:,:,2] = out_image[:,:,2]*Random_R_Scale

    # brightness and saturation
    Random_Saturation_Scale = random.randint(50,130)*0.01
    Random_Value_Scale = random.randint(50,200)*0.01

    hsv = cv2.cvtColor(out_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    Saturation_New = cv2.multiply(s,Random_Saturation_Scale)
    Value_New = cv2.multiply(v,Random_Value_Scale)
    
    Final_hsv = cv2.merge((h, Saturation_New, Value_New))
    out_image= cv2.cvtColor(Final_hsv, cv2.COLOR_HSV2BGR)
    
    
# ------------ Save image ---------------------
    cv2.imwrite(Output_Path+ Paths_To_Sets[Set_Index] + str(progress)+ "_"+ Name_of_Obj[Obj_Index] + ".jpg",out_image)

# ------------ Creating annotation file ---------------------#
    Annotation_file = open(Output_Path + Paths_To_Sets[Set_Index] + str(progress) + "_" + Name_of_Obj[Obj_Index]+".txt","w+")
    Annotation_file.seek(0)
    Width = float(topx-bottomx)/Size_Of_out_Image
    Width = float(topx-bottomx)/Size_Of_out_Image + 0.008
    Height = float(topy-bottomy)/Size_Of_out_Image + 0.008
    Center_y = float((topx+bottomx+2*Offset_x)/2.0)/Size_Of_out_Image
    Center_x = float((topy+bottomy+2*Offset_y)/2.0)/Size_Of_out_Image
    Annotation_file.write( str(Obj_Index) + " " + str(Center_y) + " " + str(Center_x)+ " " + str(Width) + " " + str(Height) + "\n")
    Annotation_file.close                   #The notation of x before is changed due to how opencv works with axis
    

# ------------ Show image and bounding box ---------------------#
    """
    # Blue color in BGR 
    color1 = (255, 0, 0) 
    color2 = (0, 0, 255)
    color3 = (0, 255, 0) 
    color4 = (255, 0, 255) 
  
    #Line thickness of 2 px 
    thickness = 1

    Height = float(topx-bottomx)#/Size_Of_out_Image
    Width = float(topy-bottomy)#/Size_Of_out_Image
    Center_y = float(Height/2+bottomy+Offset_y)#/Size_Of_out_Image
    Center_x = float(Width/2+bottomx+Offset_x)#/Size_Of_out_Image

    print("Height: " + str(Height))
    print("Width: " + str(Width))
    print("Center x: " + str(Center_y))
    print("Center y: " + str(Center_x))

    cv2.rectangle(out_image, (bottomy+Offset_y,bottomx+Offset_x),(topy+Offset_y,topx+Offset_x), color1, thickness)
    cv2.circle(out_image, (int(Center_y),int(Center_x)), 2, color2, -1) 
    cv2.circle(out_image, (bottomy+Offset_y,bottomx+Offset_x), 2, color3, thickness) 
    cv2.circle(out_image, (topy+Offset_y,topx+Offset_x), 2, color4, thickness) 

    cv2.imshow("Output",out_image)
    cv2.waitKey(0)
    """
# ------------ Create set file ---------------------#
    Set_file = open(Output_Path +Set_name[Set_Index]+".txt","a+")
    Set_file.write( "Data/" + Set_name[Set_Index] +"/" + str(progress) + "_" + Name_of_Obj[Obj_Index] + ".jpg"+"\n")
    Set_file.close


# ------------ Show Image ---------------------#
    #cv2.imshow('Output', out_image)
    #cv2.waitKey(100)
    #cv2.destroyAllWindows()

print(Test1)
print(Test)

print ('Generating detection training images took ', time.time()-start, 'seconds.')
