import os
import argparse
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.ed.minibatch.build_minibatch import BuildMinibatch


class DataLoader(object):
    def __init__(self, args):
        self.args = args

        if hasattr(args, 'build_minibatch'):
            self.build_minibatch = args.build_minibatch
        else:
            self.build_minibatch = args.build_minibatch =  BuildMinibatch(args)

        self.train_file = os.path.join(args.root_data_dir, 'generated/test_train_data/aida_train.csv')
        self.reader = open(self.train_file, 'r')

        # **YD** args.store_train_data has been implemented, RAM by default
        print('==> Loading training data with option ' + args.store_train_data)

        self.doc2id = None
        self.id2doc = None
        self.all_docs_inputs = None
        self.all_docs_targets = None
        self.all_doc_lines = None
        self.load_doc()

    def one_doc_to_minibatch(self, doc_lines):
        # -- Create empty mini batch:
        num_mentions = len(doc_lines)
        assert num_mentions > 0

        # **YD** "empty_minibatch_with_ids" has been implemented
        inputs = self.build_minibatch.empty_minibatch_with_ids(num_mentions)
        targets = torch.zeros(num_mentions, dtype=torch.long) * (-1)

        for i in range(num_mentions):

            # **YD** "process_one_line" has been implemented
            target = self.build_minibatch.process_one_line(doc_lines[i], inputs, i, True)

            # **YD** check bug position 1
            assert inputs[1][0].size(1) == self.args.num_cand_before_rerank

            targets[i] = target
            assert target >= 0

        # **YD** check bug position 1 - 1
        assert inputs[1][0].size(1) == self.args.num_cand_before_rerank

        return inputs, targets

    def load_doc(self):

        # **YD** "args.store_train_data" has been implemented
        if self.args.store_train_data == 'RAM':
            all_docs_inputs = dict()
            all_docs_targets = dict()
            doc2id = dict()
            id2doc = dict()
            cur_doc_lines = []
            prev_doc_id = -1

            for line in self.reader:
                line = line.rstrip()
                parts = line.split('\t')
                doc_name = parts[0]
                if doc_name not in doc2id:
                    if prev_doc_id > -1:
                        inputs, targets = self.one_doc_to_minibatch(cur_doc_lines)

                        # **YD** "minibatch_table2tds" unnecessary

                        all_docs_inputs[prev_doc_id] = inputs
                        all_docs_targets[prev_doc_id] = targets

                    cur_docid = len(doc2id)
                    id2doc[cur_docid] = doc_name
                    doc2id[doc_name] = cur_docid
                    cur_doc_lines = []
                    prev_doc_id = cur_docid

                cur_doc_lines.append(line)

            if prev_doc_id > -1:
                inputs, targets = self.one_doc_to_minibatch(cur_doc_lines)

                # **YD** "minibatch_table2tds" unnecessary

                all_docs_inputs[prev_doc_id] = inputs
                all_docs_targets[prev_doc_id] = targets

            assert len(doc2id) == len(all_docs_inputs), str(len(doc2id)) + ' ' + str(len(all_docs_inputs))
            self.doc2id = doc2id
            self.id2doc = id2doc
            self.all_docs_inputs = all_docs_inputs
            self.all_docs_targets = all_docs_targets

        else:
            all_doc_lines = dict()
            doc2id = dict()
            id2doc = dict()

            for line in self.reader:
                line = line.rstrip()
                parts = line.split('\t')
                doc_name = parts[0]

                if doc_name not in doc2id:
                    cur_docid = len(doc2id)
                    id2doc[cur_docid] = doc_name
                    doc2id[doc_name] = cur_docid

                    all_doc_lines[cur_docid] = []

                all_doc_lines[doc2id[doc_name]].append(line)

            assert len(doc2id) == len(all_doc_lines)

            self.doc2id = doc2id
            self.id2doc = id2doc
            self.all_doc_lines = all_doc_lines

        self.reader.close()

    def get_minibatch(self):
        # inputs, targets = None, None
        if self.args.store_train_data == 'RAM':
            # **YD** for debug case, set the random_docid a constant
            random_docid = random.randrange(0, len(self.id2doc))
            # random_docid = 2
            # **YD** "minibatch_tds2table" unnecessary
            inputs = copy.deepcopy(self.all_docs_inputs[random_docid])
            targets = copy.deepcopy(self.all_docs_targets[random_docid])

        else:
            doc_lines = self.all_doc_lines[random.randrange(0, len(self.id2doc))]
            inputs, targets = self.one_doc_to_minibatch(doc_lines)

        # **YD** check bug position 2
        # print(inputs[1][0].size())
        assert inputs[1][0].size(1) == self.args.num_cand_before_rerank

        inputs, targets = self.build_minibatch.minibatch_to_correct_type(inputs, targets, True)

        return inputs, targets


"""
# **YD** not sure the implementation, seems to be unnecessary
def minibatch_tds2table(inputs_tds):
    inputs = [[], [], []]

    inputs[0].append(inputs_tds[0][0])
    inputs[1].append(inputs_tds[1][0])
    inputs[2].append(inputs_tds[2])
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parser for ed model',
        allow_abbrev=False,
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )

    parser.add_argument(
        '--ent_vecs_filename',
        type=str,
        # default='ent_vecs__ep_231.pt',
        required=True,
        help='File name containing entity vectors generated with entities/learn_e2v/learn_a.py',
    )
    parser.add_argument(
        '--word_vecs_size',
        type=int,
        default=300,
        help='dimension of word embedding',
    )

    parser.add_argument(
        '--ent_vecs_size',
        type=int,
        default=300,
        help='dimension of entity embedding',
    )

    parser.add_argument(
        '--word_vecs',
        type=str,
        default='w2v',
        choices=['glove', 'w2v'],
        help='300d word vectors type: glove | w2v',
    )

    parser.add_argument(
        '--unig_power',
        type=float,
        default=0.6,
        help='Negative sampling unigram power (0.75 used in Word2Vec).',
    )

    parser.add_argument(
        '--entities',
        type=str,
        default='RLTD',
        choices=['RLTD', 'ALL'],
        help='Set of entities for which we train embeddings: '
             ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
    )

    parser.add_argument(
        '--store_train_data',
        type=str,
        default='RAM',
        choices=['RAM', 'DISK'],
        help='Where to read the training data from, RAM to put training instances in RAM, which has enought space'
             'to store aida-train dataset',
    )

    args = parser.parse_args()

    args.ctxt_window = 100
    args.num_cand_before_rerank = 30
    args.keep_e_ctxt = 3
    args.keep_p_e_m = 4
    args.max_num_cand = args.keep_p_e_m + args.keep_e_ctxt

    data_loader = DataLoader(args)
    for _ in range(100):
        inputs, targets = data_loader.get_minibatch()
        print("targets", targets)
