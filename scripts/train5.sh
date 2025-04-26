#!/bin/bash

available_gpus=(5)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  python /export/home/sheid/SD_Finetuning/dmt/train.py data=laion experiment=exp_f_ni_d_7_r_0
done
wait
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  python /export/home/sheid/SD_Finetuning/dmt/train.py data=laion experiment=exp_f_ni_d_7_r_1
done