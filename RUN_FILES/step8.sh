#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu # specify the queue
#$-l gpu_card=1
#$-N CRC_step8

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/deep_ed_PyTorch/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/deep_ed_PyTorch/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/deep_ed_PyTorch/data

python3 ${CODE_DIR}/entities/relatedness/filter_wiki_canonical_words_RLTD.py --root_data_dir ${DATA_PATH}
python3 ${CODE_DIR}/entities/relatedness/filter_wiki_hyperlink_contexts_RLTD.py --root_data_dir ${DATA_PATH}