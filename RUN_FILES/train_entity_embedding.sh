#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu@qa-xp-004 # specify the queue
#$-l gpu_card=4
#$-N CRC_train_ent_embedding

export PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/lib:$LD_LIBRARY_PATH

CODE=/scratch365/yding4/EL_resource/baseline/deep_ed_PyTorch/deep_ed_PyTorch/entities/learn_e2v/learn_a.py

python ${CODE}
