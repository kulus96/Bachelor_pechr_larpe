"""
Taking rendered images and adding a background and anotation for YOLOv3
Can take multiple rendered objects and add to same background.
The objects overlaps a bit.
Generates annotation for the object
Author:lrpd/larpe17
24-02-2020
"""

import cv2
import random
from PIL import Image
import numpy as np
import os
import time
import progressbar
import shutil
import renderTestData

Paths_To_Rendered_Images = []
Paths_To_Background_Images = []
Name_of_Obj = []
Paths_To_3D_Models = []
Paths_To_Sets = ["Train/","Validation/","Test/"]
Set_name =["Train","Validation","Test"]
Camera_Distance = []

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
        
        if title == 'Amount':
            Number_of_render_images = int(line)


        elif title == '3D':
            Path_To_3D_Model = line.rstrip()
            Model = os.listdir(Path_To_3D_Model)
            for file in Model:
                Paths_To_3D_Models.append(Path_To_3D_Model+file)
        
        elif title == 'Name':
            Name_of_Obj.append(line.rstrip())
        
        elif title == 'Distance':
            Camera_Distance.append(int(line))

        elif title == 'Rendered':
            Path_To_Images_rendered = line.rstrip()
            if (os.path.exists(Path_To_Images_rendered)):
                shutil.rmtree(Path_To_Images_rendered)
            os.mkdir(Path_To_Images_rendered)
            for i in Name_of_Obj:
                os.mkdir(Path_To_Images_rendered + "/"+ i)

        elif title == 'Background':
            Path_To_Images_Background = line.rstrip()
            bg = os.listdir(Path_To_Images_Background)
            for file in bg:
                Paths_To_Background_Images.append(Path_To_Images_Background+file)

        elif title == 'Number':
            Number_Of_repetition = int(line)

        elif title == 'Training':
            Output_Path = line.rstrip()
            if (os.path.exists(Output_Path)):
                shutil.rmtree(Output_Path)
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
start = time.time()  

if not os.path.exists(Output_Path+Paths_To_Sets[0]):
    os.mkdir(Output_Path+Paths_To_Sets[0])

if not os.path.exists(Output_Path+Paths_To_Sets[1]):
    os.mkdir(Output_Path+Paths_To_Sets[1])    

if not os.path.exists(Output_Path+Paths_To_Sets[2]):
    os.mkdir(Output_Path+Paths_To_Sets[2])
# ----------- Start of program render images of each object ------------
for i in range(len(Name_of_Obj)):
    print("Render images of "+Name_of_Obj[i] + " from different angles")
    for progress in progressbar.progressbar(range(Number_of_render_images)):
	    renderTestData.renderObj(Paths_To_3D_Models[i],Path_To_Images_rendered + "/"+Name_of_Obj[i]+ "/",Name_of_Obj[i]+"_"+str(i)+"_"+str(progress),Camera_Distance[i],int(random.uniform(0,360)),int(random.uniform(0,360)),0,int(random.uniform(0,360)),int(random.uniform(0,360)),int(random.uniform(0,360)),0,0,0,100*0.01,100*0.01 ,random.uniform(35,100)*0.01)#random.uniform(0,100)*0.01,random.uniform(0, 100)*0.01,random.uniform(0,100)*0.01)

for i in range(len(Name_of_Obj)):    
    p = os.listdir(Path_To_Images_rendered+Name_of_Obj[i])
    Paths_To_Rendered_Images.append([])
    for file in p:
        Paths_To_Rendered_Images[i].append(Path_To_Images_rendered+Name_of_Obj[i]+'/'+file)

# ----------- Start of program to make trainging data ------------            
Number_Of_images = 0
Obj_Index = 0
Number_Of_images_In_Set = [Number_Of_repetition*(Size_Of_Training_Set*0.01),Number_Of_repetition*(Size_Of_Validation_Set*0.01),Number_Of_repetition*(Size_Of_Test_Set*0.01)]
Set_Index = 0


print ("Generating " + str(Number_Of_repetition) + " images")
for progress in progressbar.progressbar(range(Number_Of_repetition)):
    random.seed(a=None)
    Number_Of_images = Number_Of_images + 1

    if Number_Of_images_In_Set[Set_Index] == Number_Of_images and not (Set_Index == 2):
        Set_Index = Set_Index + 1
        Number_Of_images = 0
    
    Number_Of_Objects_In_Image = int(random.uniform(1,4))
    out_image = cv2.imread(Paths_To_Background_Images[int(random.uniform(0,len(Paths_To_Background_Images)-1))])
    out_image = cv2.resize(out_image,(Size_Of_out_Image,Size_Of_out_Image),interpolation=cv2.INTER_CUBIC)
   
    Data_On_Objects = []
    Annotation_file = open(Output_Path + Paths_To_Sets[Set_Index] + str(progress) + "_" + str(Number_Of_Objects_In_Image) +".txt","w+")
    Annotation_file.close

    for h in range(Number_Of_Objects_In_Image):
        
        # ----------- Random object and background ------------
        Obj_Index =int(random.uniform(0,np.shape(Paths_To_Rendered_Images)[0]))
        Image_Index = int(random.uniform(0,np.shape(Paths_To_Rendered_Images)[1]))

        # ----------- Loading the images -----------
        rend_image = cv2.imread(Paths_To_Rendered_Images[Obj_Index][Image_Index])

        # ----------- Resizing the image ------------
        (Size_Of_Input_image_y,Size_Of_Input_image_x,Output_Channels) = rend_image.shape
        Scale = random.uniform(1.2,0.5)
        rend_image = cv2.resize(rend_image,(int(Size_Of_Input_image_y*Scale),int(Size_Of_Input_image_x*Scale)))

        # ----------- Finding the contour of the object in the image ------------
        gray_image = cv2.cvtColor(rend_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_image, 10, 10, 10)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        mask = np.zeros_like(rend_image) # Create mask where white is what we want, blackotherwise
        cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask

        (y, x, BGR) = np.where(mask == 255)


        # ----------- Finding a random possition for the object in the output image ------------
        (bottomy, bottomx) = (np.min(y), np.min(x))
        (topy, topx) = (np.max(y), np.max(x))
    
        Object_fits = False
        Offset_x = 0
        Offset_y = 0
        (Size_Of_Output_image_y,Size_Of_Output_image_x,Output_Channels) = out_image.shape
        
        
        Overlap = 0.9 #To what degree an object can overlap another. Set to 1, if the objects are not allowed to overlap
        Number_Of_Trials = 0

        while(not Object_fits):
            Offset_x = int(random.uniform(-Size_Of_Output_image_x/2, Size_Of_Output_image_x/2))
            Offset_y = int(random.uniform(-Size_Of_Output_image_y/2, Size_Of_Output_image_y/2))
            if topx+Offset_x >= 0 and topy+Offset_y >= 0 :
                if bottomx+Offset_x > 0 and bottomy+Offset_y > 0 and topx+Offset_x < Size_Of_Output_image_x and topy+Offset_y < Size_Of_Output_image_y :
                    if not(len(Data_On_Objects) == 0):
                        Object_fits = True
                        for g in range(len(Data_On_Objects)):
                            if  not(bottomx+Offset_x > Data_On_Objects[g][1]*Overlap or topx+Offset_x < Data_On_Objects[g][0]*Overlap):
                                if not ( bottomy+Offset_y>Data_On_Objects[g][3]*Overlap or topy+Offset_y < Data_On_Objects[g][2]*Overlap):
                                    Object_fits = False
                    else:
                        Object_fits = True
            Number_Of_Trials = Number_Of_Trials + 1
            if Number_Of_Trials == 10000:
                break
        if Object_fits == False:
            break
        Data_On_Objects.append([bottomx+Offset_x,topx+Offset_x,bottomy + Offset_y,topy + Offset_y])

        # ----------- Crop the image and copying it into the background ------------
        for l in range(len(x)):
            if not(np.any(rend_image[y[l],x[l]]) == 0):
                out_image[y[l]+Offset_y,x[l]+Offset_x]=rend_image[y[l],x[l]]
            # ------------ Creating annotation file ---------------------#
        Annotation_file = open(Output_Path + Paths_To_Sets[Set_Index] + str(progress) + "_" + str(Number_Of_Objects_In_Image)+".txt","a+")
        Annotation_file.seek(0)
        Width = float(topx-bottomx)/Size_Of_out_Image + 0.008
        Height = float(topy-bottomy)/Size_Of_out_Image + 0.008
        Center_y = float((topx+bottomx+2*Offset_x)/2.0)/Size_Of_out_Image
        Center_x = float((topy+bottomy+2*Offset_y)/2.0)/Size_Of_out_Image

        Annotation_file.write( str(Obj_Index) + " " + str(Center_y) + " " + str(Center_x)+ " " + str(Width) + " " + str(Height) + "\n")
        Annotation_file.close                       

    # ------------ Show image and bounding box ---------------------#
        """
        # Blue color in BGR 
        color1 = (255, 0, 0) 
        color2 = (0, 0, 255)
        color3 = (0, 255, 0) 
        color4 = (255, 0, 255) 
    
        #Line thickness of 2 px 
        thickness = 1

        Width = float(topx-bottomx)
        Height = float(topy-bottomy)
        Center_y = float((topx+bottomx+2*Offset_x)/2.0)
        Center_x = float((topy+bottomy+2*Offset_y)/2.0)

        
        print(Size_Of_out_Image)
        print("Height: " + str(Height/Size_Of_out_Image))
        print("Width: " + str(Width/Size_Of_out_Image))
        print("Center x: " + str(Center_y))
        print("Center y: " + str(Center_x))
        print("Center x2: " + str(float(topy+bottomy)/2))
        print("Center y2: " + str(float(topx+bottomx)/2))

        cv2.rectangle(out_image, (int(Center_y-Width/2),int(Center_x-Height/2)),(int(Center_y+Width/2),int(Center_x+Height/2)), color1, thickness)
        cv2.circle(out_image, (int(Center_y),int(Center_x)), 8, color2, -1) 
        cv2.circle(out_image, (int(Center_y-Width/2),int(Center_x-Height/2)), 2, color3, thickness) 
        cv2.circle(out_image, (int(Center_y+Width/2),int(Center_x+Height/2)), 2, color4, thickness) 
        cv2.line(out_image,(int(Center_y-Width/2),int(Center_x-Height/2)),(int(Center_y+Width/2),int(Center_x+Height/2)),color3,thickness)
        cv2.line(out_image,(int(Center_y+Width/2),int(Center_x+Height/2)),(int(Center_y-Width/2),int(Center_x-Height/2)),color3,thickness)

        cv2.imshow("Output",out_image)
        cv2.waitKey(0)
        
"""
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

    # Adding gausion noise
    Degree_Of_Blur = random.randint(0,6)
    if not(Degree_Of_Blur==6):
        out_image = cv2.GaussianBlur(out_image,(5,5),Degree_Of_Blur)
        
        
# ------------ Save image ---------------------
    cv2.imwrite(Output_Path+ Paths_To_Sets[Set_Index] + str(progress)+ "_"+ str(Number_Of_Objects_In_Image) + ".jpg",out_image)

# ------------ Create set file ---------------------#
    Set_file = open(Output_Path +Set_name[Set_Index]+".txt","a+")
    Set_file.write( "Data/" + Set_name[Set_Index] +"/" + str(progress) + "_" + str(Number_Of_Objects_In_Image) + ".jpg"+"\n")
    Set_file.close


# ------------ Show Image ---------------------#
    """
    cv2.imshow('Output', out_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    """

print ('Generating detection training images took ', time.time()-start, 'seconds.')
