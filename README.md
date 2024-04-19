# MaskLRF
## Introduction
This repository provides the official codes (PyTorch implementation) for the paper **["MaskLRF: Self-supervised Pretraining via Masked Autoencoding of Local Reference Frames for Rotation-invariant 3D Point Set Analysis"](https://arxiv.org/abs/2403.00206)**. The paper is currently under review.
## Pre-requisites
My code has been tested on Ubuntu 22.04. I highly recommend using the Docker container "nvcr.io/nvidia/pytorch:21.09-py3", which is provided by [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags). 
After launching the Docker container, run the following shell script to install the prerequisite libraries.
```
./prepare.sh
```
## Datasets
See [DATASET.md](DATASET.md) for details. 
## Self-supervised pretraining by MaskLRF
Run the following shell script to start pretraining from scratch with the configurations used in the paper.
```
./Run_MaskLRF_pretraining.sh
```
You can also download the pretrained DNN parameters below. 
| DNN model | Dataset for pretraining | URL |
| ---- | ---- | ---- |
| R2PT | ShapeNetCore55 | aaa | 

## Supervised finetuning
### Real-world object classification
TODO
### Few-shot object classification
TODO
### Part segmentation
TODO
