#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-006 # specify the queue
#$-l gpu_card=0
#$-N CRC_eval_local_model

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/deep_ed_PyTorch/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/deep_ed_PyTorch/data

#CODE=/scratch365/yding4/EL_resource/baseline/deep_ed_PyTorch/deep_ed_PyTorch/ed/test/test_one_loaded_model.py

export CUDA_VISIBLE_DEVICES=2
python ${CODE_DIR}/ed/test/test_one_loaded_model.py  \
    --root_data_dir ${DATA_PATH}    \
    --model_type 'local'  \
    --max_epoch 400  \
    --ent_vecs_filename 'ent_vecs__ep_51.pt' \
    --test_one_model_file 'local_99.pt'
