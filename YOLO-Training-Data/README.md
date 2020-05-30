### Contents
##1.3DToTrain
#Requirements
- Added 3D models to the "3DModel" folder (.obj or .ply).
- Added background images to the "BackgroundImages" folder.
- Configurations can be made in the file "createData.cfg".
#Flow of the program
- Render images with no background from CAD models.
- Using the rendered images and adding background.
- Augment the images.
- Making annotation file.
- Dividing all the images in to sets (Training, validation, and test).

##2.AugmentRealImages
#Requirements
- Added images and annotation files to the folder "Images".
- Changes in configurations can be made in the "AugmentRealImages.py" file.
#Flow of the program
- Taking real images with annotations.
- Making duplicates with augmentations.
- Dividing into sets.

##3.CombiningImages
#Requirements
- Added images to the folder 
- Configurations can be made in the "CombingImagesInSets.py" file.
#Flow of the program
- Annotations and images needs to be ready.
- Making sets of the images and the corresponding annotation files.
- Making the annotation file list.
