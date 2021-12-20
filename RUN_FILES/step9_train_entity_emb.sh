#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-titanx-001 # specify the queue
#$-l gpu_card=0
#$-N CRC_step9_entity_embedding

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/deep_ed_PyTorch/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/deep_ed_PyTorch/data

export CUDA_VISIBLE_DEVICES=3
python3 ${CODE_DIR}/entities/learn_e2v/learn_a.py --max_epoch 200 --root_data_dir ${DATA_PATH}
