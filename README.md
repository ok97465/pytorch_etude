# pytorch_etude

The repository is for deep learning example codes using pytorch, pytorch_lighting, and tensorboard.

Examples log in the tensorboard. Some examples provide the script file to demostrate the result of model.

## Ex1 Linear Model

In this example, Linear model guesses the slope and bias of the noisy input.

<img src="/doc/Ex01.png" width="400"/>

## Ex2 Linear Model for MNIST

The model of ex2 consists of linear modules. This model classifies the handwritten image of MNIST.

|           Plot Good Result            |     Plot  Bad Result           |      Confusion Matrix     |      Vector visualization     |
|-----------------------------|---------------------------|---------------------------|---------------------------|
|![Ex2 Good](/doc/Ex02Good.png)|![Ex2 Bad](/doc/Ex02Bad.png)|![Ex2 CM](/doc/Ex02CM.png)  |![Ex2 vector visualization](/doc/Ex02Tsne.gif) |

## Ex3 CNN Model for MNIST

Ex3 has higher accuracy than Ex2.

|    Output of 1st Conv2D    |        Output of 2nd Conv2D        |
|---------------------------------------|--------------------------------|
|![Ex3 1st Output](/doc/Ex03Conv2d1st.png)|![Ex3 2nd Output](/doc/Ex03Conv2d2nd.png)|

## EX4 Denoise Autoencoder for Denoising

The model of ex4 consists of convolutional networks. This model removes noise from the mnist images to which noise has been added.

|    Traninig process in Tensorboard    |        Denoising Result        |
|---------------------------------------|--------------------------------|
|![Ex4 process](/doc/Ex04Training.png)|![Ex4 Denoise](/doc/Ex04Denoise.png)|

## EX5 CAM(Class Activation Map) for Mnist

This model use GAP(Global Average Pool) to get CAM.

|    CAM of good result    |        CAM of bad result        |
|---------------------------------------|--------------------------------|
|![Ex5 Good](/doc/Ex05Good.png)|![Ex5 Bad](/doc/Ex05Bad.png)|

## EX6 GradCAM(Class Activation Map) for Mnist

This model use GradCAM to get CAM.
(GradCAM code from [1Konny](<https://github.com/1Konny/gradcam_plus_plus-pytorch>))

|    GradCAM of good result    |        GradCAM of bad result        |
|---------------------------------------|--------------------------------|
|![Ex6 Good](/doc/Ex06Good.png)|![Ex6 Bad](/doc/Ex06Bad.png)|

## EX7 Interpretation model using CAPTUM

This example interpret the model using CAPTUM.

![Ex7 Bad](/doc/Ex07Bad.png)
![Ex7 Good](/doc/Ex07Good.png)

## EX8 R-CNN Model for kaggle bus truck data

This model use R-CNN to detect and classify Kaggle bus-truck data.
To reduce complexity, this model use a linear model to classify items instead of SVM in the original paper.

|  Selective Search Result   |
|----------------------------|
|![SS Result](/doc/Ex08SS.png)|

|          RCNN    Result        |
|--------------------------------|
|![RCNN Result](/doc/Ex08RCNN.png)|

## EX9 Fast R-CNN Model for kaggle bus truck data

This model contains RoiPooling to implement fast RCNN.

|           Fast   RCNN   Result           |
|------------------------------------------|
|![Fast RCNN Result](/doc/Ex09FastRCNN.png)|

## EX10 Faster R-CNN Model for kaggle bus truck data

This model use the faster rcnn in torchvision.

|           Fast   RCNN   Result           |
|------------------------------------------|
|![Faster RCNN Result](/doc/Ex10FasterRCNN.png)|

## Reference

[1] Modern Computer Vision with PyTorch: Explore deep learning concepts and implement over 50 real-world image applications.
