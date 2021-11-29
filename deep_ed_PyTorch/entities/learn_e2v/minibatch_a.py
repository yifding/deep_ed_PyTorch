import random

import torch

from deep_ed_PyTorch.words.w2v import W2V
from deep_ed_PyTorch.words.w_freq import WFreq
from deep_ed_PyTorch.entities.EX_wiki_words import ExWikiWords
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID


class MinibatchA(object):
    def __init__(self, args):
        self.args = args

        """
        self.ex_wiki_words = ExWikiWords()
        self.ent_name_id = EntNameID(args)
        self.w_freq = WFreq(args)
        self.w2v = W2V(args)
        """

        if hasattr(args, 'ex_wiki_words'):
            self.ex_wiki_words = args.ex_wiki_words
        else:
            self.ex_wiki_words = args.ex_wiki_words = ExWikiWords()

        if hasattr(args, 'ent_name_id'):
            self.ent_name_id = args.ent_name_id
        else:
            self.ent_name_id = args.ent_name_id = EntNameID(args)

        if hasattr(args, 'w2v'):
            self.w2v = args.w2v
        else:
            self.w2v = args.w2v = W2V(args)

        if hasattr(args, 'w_freq'):
            self.w_freq = args.w_freq
        else:
            self.w_freq = args.w_freq = WFreq(args)

    def empty_minibatch(self):
        # **YD** self.unk_w_id has been implemented
        ctxt_word_ids = torch.ones(
            self.args.batch_size,
            self.args.num_words_per_ent,
            self.args.num_neg_words,
            dtype = torch.long,
        ).mul(int(self.w_freq.unk_w_id))

        ent_component_words = torch.ones(self.args.batch_size, self.args.num_words_per_ent, dtype=torch.long)
        ent_wikiids = torch.ones(self.args.batch_size, dtype=torch.long)
        ent_thids = torch.ones(self.args.batch_size, dtype=torch.long)
        return [
            [ctxt_word_ids, None, None],
            [ent_component_words],
            [ent_thids, ent_wikiids],
                ]

    # -- Get functions:
    def get_pos_and_neg_w_ids(self, minibatch):
        return minibatch[0][0]

    def get_pos_and_neg_w_vecs(self, minibatch):
        return minibatch[0][1]

    def get_pos_and_neg_w_unig_at_power(self, minibatch):
        return minibatch[0][2]

    def get_ent_wiki_w_ids(self, minibatch):
        return minibatch[1][0]

    def get_ent_wiki_w_vecs(self, minibatch):
        return minibatch[1][1]

    def get_ent_thids_batch(self, minibatch):
        return minibatch[2][0]

    def get_ent_wikiids(self, minibatch):
        return minibatch[2][1]

    # -- Fills in the minibatch and returns the grd truth word index per each example.
    # -- An example in our case is an entity, a positive word sampled from \hat{p}(e|m)
    # -- and several negative words sampled from \hat{p}(w)^\alpha.

    def process_one_line(self, line, minibatch, mb_index):
        if self.args.entities == '4EX':
            ent_name = self.ex_wiki_words.ent_names_4EX[
                random.randint(low=0, high=len(self.ex_wiki_words.ent_names_4EX) - 1)
            ]
            line = self.ex_wiki_words.ent_lines_4EX[ent_name]

        parts = line.split('\t')
        num_parts = len(parts)

        if num_parts == 3:  # ---------> Words from the Wikipedia canonical page
            ent_wikiid = int(parts[0])
            words_plus_stop_words = parts[2].split(' ')

        else:   # --------> Words from Wikipedia hyperlinks
            assert num_parts >= 9, line + ' --> ' + str(num_parts)
            assert parts[5] == 'CANDIDATES', line

            last_part = parts[num_parts - 1]
            ent_str = last_part.split(',')
            ent_wikiid = int(ent_str[1])    # wikiid for the GT entity

            words_plus_stop_words = []
            left_ctxt_w = parts[3].split(' ')       # left context words

            words_plus_stop_words += left_ctxt_w[-self.args.hyp_ctxt_len:]

            right_ctxt_w = parts[4].split(' ')
            words_plus_stop_words += right_ctxt_w[:self.args.hyp_ctxt_len]

        # **YD** "get_thid", "get_wikiid_from_thid" has been implemented
        assert ent_wikiid >= 1

        ent_thid = self.ent_name_id.get_thid(ent_wikiid)
        assert (self.ent_name_id.get_wikiid_from_thid(ent_thid) == ent_wikiid)

        # **YD** "get_ent_thids_batch", has been implemented
        self.get_ent_thids_batch(minibatch)[mb_index] = ent_thid
        assert self.get_ent_thids_batch(minibatch)[mb_index] == ent_thid

        self.get_ent_wikiids(minibatch)[mb_index] = ent_wikiid

        # -- Remove stop words from entity wiki words representations.

        # **YD** "contains_w" has been implemented
        positive_words_in_this_iter = [w for w in words_plus_stop_words if self.w_freq.contains_w(w)]

        # -- Still empty ? Get some random words then.
        # **YD** "get_word_from_id", "random_unigram_at_unig_power_w_id" has been implemented
        if len(positive_words_in_this_iter) == 0:
            positive_words_in_this_iter.append(
                self.w_freq.get_word_from_id(
                    self.w_freq.random_unigram_at_unig_power_w_id()
                )
            )

        num_positive_words_this_iter = len(positive_words_in_this_iter)

        targets = torch.zeros(self.args.num_words_per_ent, dtype=torch.long)

        # -- Sample some negative words:
        # **YD** "random_unigram_at_unig_power_w_id" has been implemented, whole structure requires debugging
        self.get_pos_and_neg_w_ids(minibatch)[mb_index].apply_(
            lambda x: self.w_freq.random_unigram_at_unig_power_w_id()
        )

        # -- Sample some positive words:
        # **YD** "get_id_from_word" has been implemented
        for i in range(self.args.num_words_per_ent):
            positive_w = positive_words_in_this_iter[random.randint(0, num_positive_words_this_iter - 1)]
            positive_w_id = self.w_freq.get_id_from_word(positive_w)

            # -- Set the positive word in a random position. Remember that index (used in training).
            # **YD** "get_ent_wiki_w_ids" not implemented
            grd_trth = random.randint(0, self.args.num_neg_words - 1)
            targets[i] = grd_trth

            self.get_ent_wiki_w_ids(minibatch)[mb_index][i] = positive_w_id
            assert self.get_ent_wiki_w_ids(minibatch)[mb_index][i] == positive_w_id

            # **YD** "get_pos_and_neg_w_ids" has been implemented
            self.get_pos_and_neg_w_ids(minibatch)[mb_index][i][grd_trth] = positive_w_id

        return targets

    # -- Fill minibatch with word and entity vectors:
    # **YD** "get_pos_and_neg_w_ids" has been implemented
    def postprocess_minibatch(self, minibatch, targets):

        minibatch[0][0] = self.get_pos_and_neg_w_ids(minibatch).view(
            self.args.batch_size * self.args.num_words_per_ent * self.args.num_neg_words)

        # **YD** "get_ent_wiki_w_ids" has been implemented
        minibatch[1][0] = self.get_ent_wiki_w_ids(minibatch).view(self.args.batch_size * self.args.num_words_per_ent)

        # -- ctxt word vecs
        # **YD** "w2v.lookup_w_vecs" has been implemented
        minibatch[0][1] = self.w2v.lookup_w_vecs(self.get_pos_and_neg_w_ids(minibatch))
        minibatch[0][2] = torch.zeros(self.args.batch_size * self.args.num_words_per_ent * self.args.num_neg_words)

        # **YD** "get_w_unnorm_unigram_at_power" has been implemented
        """
        minibatch[0][2].map_(
            minibatch[0][0].float(),
            lambda _, w_id: self.w_freq.get_w_unnorm_unigram_at_power(int(w_id)),
        )
        """

        for i, w_id in enumerate(minibatch[0][0]):
            minibatch[0][2][i] = self.w_freq.get_w_unnorm_unigram_at_power(int(w_id))

    # -- Convert mini batch to correct type (e.g. move data to GPU):
    def minibatch_to_correct_type(self, minibatch):
        # **YD** "correct_type" not implemented
        minibatch[0][0] = correct_type(minibatch[0][0])
        minibatch[1][0] = correct_type(minibatch[1][0])
        minibatch[0][1] = correct_type(minibatch[0][1])
        minibatch[0][2] = correct_type(minibatch[0][2])
        minibatch[2][0] = correct_type(minibatch[2][0])
