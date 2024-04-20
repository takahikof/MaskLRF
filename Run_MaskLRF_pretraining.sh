#!/bin/bash

gpuid=0 # Specifies the ID of a GPU used for training/evaluation

### Pre-training ###
exp_name_base="pretrain"
cfgfile=./cfgs/pretrain.yaml
myarg_batch_size=64
myarg_rot_train="so3" # nr (no rotation) or so3 (random rotation)
myarg_rot_test="so3"
myarg_lrf_axis1="pca"
myarg_lrf_axis2="mean"
# myarg_output_feature_type="point_only"
# myarg_output_feature_type="point_normal"
# myarg_output_feature_type="occupancy_grid"
# myarg_output_feature_type="normal_curvature"
# myarg_output_feature_type="fpfh"
myarg_output_feature_type="pod"
myarg_pod_gridsize=6
myarg_group_size=32
myarg_num_group=64
myarg_mask_ratio=0.6
myarg_trans_depth=12
myarg_attn_mode="rel_contextual"
myarg_attn_subsample_rate=0.25
myarg_attn_target="knn_dilate"
# myarg_attn_target="knn_only"
# myarg_attn_target="dilate_only"

exp_name=$exp_name_base
CUDA_VISIBLE_DEVICES=$gpuid python -u main.py --config $cfgfile --exp_name $exp_name --val_freq 20 \
                                              --myarg_batch_size $myarg_batch_size \
                                              --myarg_rot_train $myarg_rot_train --myarg_rot_test $myarg_rot_test \
                                              --myarg_lrf_axis1 $myarg_lrf_axis1 --myarg_lrf_axis2 $myarg_lrf_axis2 \
                                              --myarg_output_feature_type $myarg_output_feature_type --myarg_pod_gridsize $myarg_pod_gridsize \
                                              --myarg_trans_depth $myarg_trans_depth --myarg_group_size $myarg_group_size \
                                              --myarg_num_group $myarg_num_group --myarg_mask_ratio $myarg_mask_ratio \
                                              --myarg_attn_mode $myarg_attn_mode --myarg_attn_subsample_rate $myarg_attn_subsample_rate \
                                              --myarg_attn_target $myarg_attn_target
exit
