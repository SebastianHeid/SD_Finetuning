#!/bin/bash

available_gpus=(4)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  python /export/home/sheid/SD_Finetuning/dmt/train.py data=laion experiment=exp_ro_01_11_r23_11_ao_01_11_a2_1_finetuning
done
