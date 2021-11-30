#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu # specify the queue
#$-l gpu_card=4
#$-N CRC_step5

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/EL_resource/baseline/deep_ed_PyTorch/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data

python3 ${CODE_DIR}/data_gen/gen_test_train_data/gen_test_train_data.py --root_data_dir ${DATA_PATH}