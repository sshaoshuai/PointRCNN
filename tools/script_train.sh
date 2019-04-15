#!/bin/bash
partition=$1
node=$2

gpu_num=1

srun --mpi=pmi2 --gres=gpu:${gpu_num} \
    -p $partition -n1 \
    --ntasks-per-node=1 \
    -K -w ${node}\
    python -u train_rcnn.py ${@:3}
