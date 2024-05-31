# MaskLRF
## Introduction
This repository provides the official codes (PyTorch implementation) for the paper **["MaskLRF: Self-supervised Pretraining via Masked Autoencoding of Local Reference Frames for Rotation-invariant 3D Point Set Analysis"](https://arxiv.org/abs/2403.00206)**. The paper is accepted to the IEEE Access journal.
## Pre-requisites
My code has been tested on Ubuntu 22.04. I highly recommend using the Docker container "nvcr.io/nvidia/pytorch:21.09-py3", which is provided by [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).
After launching the Docker container, run the following shell script to install the prerequisite libraries.
```
./prepare.sh
```
## Datasets
See [DATASET.md](DATASET.md) for details.
## Self-supervised pretraining
Run the following shell script to start pretraining from scratch with the configurations used in the paper.<br>
The pretrained parameters will be saved as "experiments/pretrain/ckpt-last.pth".
```
./Run_MaskLRF_pretraining.sh
```
You can also download the pretrained DNN parameters below. <br>
Save ckpt-last.pth in the directory "experiments/pretrain/".
| DNN model | Dataset for pretraining | Pretrained parameters |
| ---- | ---- | ---- |
| R2PT | ShapeNetCore55 | [ckpt-last.pth](https://drive.google.com/file/d/1H6tVSQD0_1yKhwc_0Z_juptGfbGXSzkZ/view?usp=sharing) |

## Supervised finetuning
Run the corresponding shell script to finetune the pretrained model and evaluate its accuracy in a downstream task.<br>
By default, finetuning/evaluation is done in the NR/SO3 rotation setting.<br>
A log file will be saved in the directory "experiments/".
### Real-world object classification
```
./Run_MaskLRF_finetuning_cls.sh
```
### Few-shot object classification
```
./Run_MaskLRF_finetuning_fewshot.sh
```
### Part segmentation
```
./Run_MaskLRF_finetuning_partseg.sh
```
## Acknowledgements
My code is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE).
