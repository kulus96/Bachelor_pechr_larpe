### Requirements

- Have installed and setup the augmented autoencoder from https://github.com/DLR-RM/AugmentedAutoencoder
- Have a fully working and trained AAE
- Added different backgrounds to the folder <Background>

### Testing the AAE by rendering images of the object and trying to find the corresponding rotation

1. Generate a random rotation
2. Render a image of the object with the random rotation and add the object to a random background
3. Add different kind of noise to the image
4. cut the object out of the image and resize the image to 128x128
5. Using the AAE find the rotation and compare with the groundtruth rotation.
6. Log angles and precision



