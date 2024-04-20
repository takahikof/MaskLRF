#!/bin/bash
gpuid=0 # Specifies the ID of a GPU used for training/evaluation

ckpts="experiments/pretrain/ckpt-last.pth"
exp_name_base="finetune_fewshot"
myarg_batch_size=32
myarg_rot_train="nr" # nr (no rotation) or so3 (random rotation)
myarg_rot_test="so3"
myarg_lrf_axis1="pca"
myarg_lrf_axis2="mean"
myarg_group_size=32
myarg_num_group=128
myarg_trans_depth=12
myarg_attn_mode="rel_contextual"
myarg_attn_subsample_rate=0.125 # k=16
myarg_attn_target="knn_dilate"

array_way=( 5 10 )
array_shot=( 10 20 )
array_fold=( 0 1 2 3 4 5 6 7 8 9 )

# ===============================================
for way in "${array_way[@]}"
do
  for shot in "${array_shot[@]}"
  do
    for fold in "${array_fold[@]}"
    do
      cfgfile=./cfgs/fewshot.yaml
      exp_name=$exp_name_base\_$myarg_rot_train\_$myarg_rot_test\_w$way\_s$shot\_f$fold
      echo $exp_name
      CUDA_VISIBLE_DEVICES=$gpuid python -u main.py --config $cfgfile --finetune_model --ckpts $ckpts --exp_name $exp_name \
                                                 --way $way --shot $shot --fold $fold \
                                                 --myarg_batch_size $myarg_batch_size \
                                                 --myarg_rot_train $myarg_rot_train --myarg_rot_test $myarg_rot_test \
                                                 --myarg_lrf_axis1 $myarg_lrf_axis1 --myarg_lrf_axis2 $myarg_lrf_axis2 \
                                                 --myarg_trans_depth $myarg_trans_depth --myarg_group_size $myarg_group_size \
                                                 --myarg_num_group $myarg_num_group \
                                                 --myarg_attn_mode $myarg_attn_mode --myarg_attn_subsample_rate $myarg_attn_subsample_rate \
                                                 --myarg_attn_target $myarg_attn_target
    done
  done
done
