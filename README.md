# CNNVO - Final Project for 603.661 Computer Vision
## Convolutional Neural Networks for Visual Odometry
This is a pytorch-based package for doing Visual Odometry by estimating frame-to-frame motion using deep Siamese network.

# Requirements
## Python packages :
numpy, scipy, matplotlib, opencv, torch, skimage, glob

## Kitti Dataset:
The Kitti dataset can be obtained from http://www.cvlibs.net/datasets/kitti/eval_odometry.php.
The package by default assumes that the path for the data is "../kitti_datasets".

## Python and CUDA versions
The package is written for python2.7 and CUDA.