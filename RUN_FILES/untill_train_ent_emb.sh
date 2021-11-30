#!/bin/bash

#$-m abe
#$-M yding4@nd.edu
#$-q gpu # specify the queue
#$-l gpu_card=4
#$-N CRC_total_ent_emb

export PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/y/yding4/.conda/envs/hetseq/lib:$LD_LIBRARY_PATH

CODE_DIR=/scratch365/yding4/EL_resource/baseline/deep_ed_PyTorch/deep_ed_PyTorch
DATA_PATH=/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data

# step1
python3 ${CODE_DIR}/data_gen/gen_p_e_m/gen_p_e_m_from_wiki.py --root_data_dir ${DATA_PATH} > step1.log

# step2
python3 ${CODE_DIR}/data_gen/gen_p_e_m/merge_crosswikis_wiki.py --root_data_dir ${DATA_PATH}

# step3
python3 ${CODE_DIR}/data_gen/gen_p_e_m/gen_p_e_m_from_yago.py --root_data_dir ${DATA_PATH}

# step4
python3 ${CODE_DIR}/entities/ent_name2id_freq/e_freq_gen.py --root_data_dir ${DATA_PATH}

# step5
python3 ${CODE_DIR}/data_gen/gen_test_train_data/gen_test_train_data.py --root_data_dir ${DATA_PATH}

# step6
python3 ${CODE_DIR}/data_gen/gen_wiki_data/gen_ent_wiki_w_repr.py --root_data_dir ${DATA_PATH}
python3 ${CODE_DIR}/data_gen/gen_wiki_data/gen_wiki_hyp_train_data.py --root_data_dir ${DATA_PATH}

# step7
python3 ${CODE_DIR}/words/w_freq/w_freq_gen.py --root_data_dir ${DATA_PATH}

# step8
python3 ${CODE_DIR}/entities/relatedness/filter_wiki_canonical_words_RLTD.py --root_data_dir ${DATA_PATH}
python3 ${CODE_DIR}/entities/relatedness/filter_wiki_hyperlink_contexts_RLTD.py --root_data_dir ${DATA_PATH}

# step9. train entity embedding
python3 ${CODE_DIR}/entities/learn_e2v/learn_a.py --max_epoch 200 --root_data_dir ${DATA_PATH}
