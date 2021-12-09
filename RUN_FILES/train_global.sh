#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-016 # specify the queue
#$-l gpu_card=0
#$-N CRC_train_global

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/deep_ed_PyTorch/data

export CUDA_VISIBLE_DEVICES=1
python ${CODE_DIR}/deep_ed_PyTorch/ed/train.py  \
    --model_type 'global'  \
    --root_data_dir ${DATA_PATH}    \
    --max_epoch 400 \
    --ent_vecs_filename 'ent_vecs__ep_51.pt'
