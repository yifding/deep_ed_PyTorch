#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu # specify the queue
#$-l gpu_card=4
#$-N CRC_train_ent_embedding

export PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/Transformer/lib:$LD_LIBRARY_PATH

CODE=/scratch365/yding4/EL_resource/baseline/deep_ed_PyTorch/deep_ed_PyTorch/ed/test/test_one_loaded_model.py

python ${CODE}  --model_type 'local'  --max_epoch 400  --test_one_model_file 'local_380.pt'