#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-016 # specify the queue
#$-l gpu_card=0
#$-N CRC_eval_ent_embedding

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/deep_ed_PyTorch/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/deep_ed_PyTorch/data
DIR=/scratch365/yding4/deep_ed_PyTorch/data/generated/ent_vecs/

export CUDA_VISIBLE_DEVICES=2
python ${CODE_DIR}/entities/learn_e2v/YD_eval_entity_embedding.py \
    --root_data_dir ${DATA_PATH}    \
    --dir ${DIR}
#    > CRC_eval_ent_embedding.txt
