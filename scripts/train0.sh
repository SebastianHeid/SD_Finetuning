#!/bin/bash

available_gpus=(7)
export OMP_NUM_THREADS=16
for gpu in "${available_gpus[@]}"; do
  export CUDA_VISIBLE_DEVICES=$gpu
  python /export/home/sheid/SD_Finetuning/dmt/train.py data=laion experiment=exp_ro_0123_1111_a012_111
done
wait
