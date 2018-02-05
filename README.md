# Face Detection and Facial Attribute Editing

This is the final project for Deep Learning for Computer Vision lecture [DL4CV](https://vision.cs.tum.edu/teaching/ss2017/dl4cv) in Technical University of Munich.

## Abstract

It is practically infeasible to collect images with arbitrarily specified attributes for faces of each person. We are interested in the problem of manipulating natural images of faces by controlling some attributes of interest. Existing approaches take as an input a cropped face and then perform facial attribute editing task. However, in the real-world scenario, the input is not a cropped face. We developed a pipeline which first detects faces in the image and then changes the specified attributes for each of the detected faces.

## Code

For this project we performed 2 main step:

1.  Face detection: we fine-tuned Tensorflow object detection SSD model. All codes are available in `src/detection`. `data.py` prepares data in the proper format for Tensorflow. Training is done using the codes of the original repository.  

2.  Facial attribute editing: we trained the network using the original code provided by the authors. All codes are available in `src/editing`.

Our full pipe-line is available at `detection_editing.ipynb`.
