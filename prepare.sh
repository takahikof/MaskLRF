#!/bin/bash

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Pytorch3D
conda update conda
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"

pip install -r requirements.txt

apt-get update
apt-get install -y libgl1-mesa-dev
apt-get install -y bc
