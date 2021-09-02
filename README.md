# Image-Inpainting
GAN based architecture for solving visual in-painting task. Demonstrated that a network with multiple discriminators and skip connections (U-Net like architecture) outperforms baseline GAN in terms of quality of generated images. You can find a detailed <a href="https://github.com/Rajeshyd0308/Image-Inpainting/blob/main/report.pdf" target="_blank">report here</a>.
## How to run the code
You can directly start by cloning the repository, but make sure you got the dataset paths right. Running the train.py will store the results after every epoch in a seperate folder. I have trained my algorithm on entire MS-COCO 2014 dataset.

# Some of the results:
![alt text](https://github.com/Rajeshyd0308/Image-Inpainting/blob/main/images/results.PNG)
# Model architecture:
It is a U-Net GAN like architecture but with 2 discriminators.
![alt text](https://github.com/Rajeshyd0308/Image-Inpainting/blob/main/images/Capture.PNG)
