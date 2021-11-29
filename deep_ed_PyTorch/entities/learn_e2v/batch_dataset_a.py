import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from minibatch_a import MinibatchA
# **YD** the logic of reading and changing requires an individual test in a separate directory
# DONE! test/two_file_load


class BatchDatasetA(object):
    def __init__(self, args):
        self.args = args
        self.mini_batch_a = MinibatchA(args)

        if args.entities == 'ALL':
            self.wiki_words_train_file = os.path.join(args.root_data_dir, 'generated/wiki_canonical_words.txt')
            self.wiki_hyp_train_file = os.path.join(args.root_data_dir, 'generated/wiki_hyperlink_contexts.csv')
        else:
            self.wiki_words_train_file = os.path.join(args.root_data_dir, 'generated/wiki_canonical_words_RLTD.txt')
            self.wiki_hyp_train_file = os.path.join(args.root_data_dir, 'generated/wiki_hyperlink_contexts_RLTD.csv')

        self.wiki_words_it = open(self.wiki_words_train_file, 'r')
        self.wiki_hyp_it = open(self.wiki_hyp_train_file, 'r')

        assert hasattr(self.args, 'num_passes_wiki_words')
        self.train_data_source = 'wiki-canonical'
        self.num_passes_wiki_words = 1

    def __del__(self):
        self.wiki_words_it.close()
        self.wiki_hyp_it.close()

    def read_one_line(self):
        if self.train_data_source == 'wiki-canonical':
            line = self.wiki_words_it.readline()
        else:
            assert self.train_data_source == 'wiki-canonical-hyperlinks'
            line = self.wiki_hyp_it.readline()

        if not line:
            if self.num_passes_wiki_words == self.args.num_passes_wiki_words:
                self.train_data_source = 'wiki-canonical-hyperlinks'
                self.wiki_words_it.close()
                print('\n\n' + 'Start training on Wiki Hyperlinks' + '\n\n')

            print('Training file is done. Num passes = ', self.num_passes_wiki_words, '. Reopening.')
            self.num_passes_wiki_words += 1

            if self.train_data_source == 'wiki-canonical':
                self.wiki_words_it = open(self.wiki_words_train_file, 'r')
                line = self.wiki_words_it.readline()
            else:
                self.wiki_hyp_it = open(self.wiki_hyp_train_file, 'r')
                line = self.wiki_hyp_it.readline()

        return line

    def patch_of_lines(self, batch_size):
        lines = []
        cnt = 0

        assert batch_size > 0

        while cnt < batch_size:
            # **YD** "read_one_line", has been implemented
            line = self.read_one_line()
            cnt += 1
            lines.append(line)

        assert len(lines) == batch_size

        return lines

    def get_minibatch(self):
        # -- Create empty mini batch:
        lines = self.patch_of_lines(self.args.batch_size)
        inputs = self.mini_batch_a.empty_minibatch()
        targets = torch.ones(self.args.batch_size, self.args.num_words_per_ent).long()

        # -- Fill in each example:
        for i in range(self.args.batch_size):
            sample_line = lines[i]
            # **YD** "process_one_line", has been implemented
            target = self.mini_batch_a.process_one_line(sample_line, inputs, i)
            targets[i] = target.clone()

        # --- Minibatch post processing:
        # **YD** "postprocess_minibatch", has been implemented
        self.mini_batch_a.postprocess_minibatch(inputs, targets)
        targets = targets.view(self.args.batch_size * self.args.num_words_per_ent)

        return inputs, targets
