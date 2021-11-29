import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.words.w2v import W2V
from deep_ed_PyTorch.words.w_freq import WFreq
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID, EFreqIndex

"""
-- Reranking of entity candidates:
---- Keeps only 4 candidates, top 2 candidates from p(e|m) and top 2 from <ctxt_vec,e_vec>
max_num_cand = opt.keep_p_e_m + opt.keep_e_ctxt
"""


class BuildMinibatch(object):
    def __init__(self, args):
        self.args = args
        self.ent_embed_file = os.path.join(args.root_data_dir, 'generated/ent_vecs/' + args.ent_vecs_filename)

        if hasattr(args, 'w2v'):
            self.w2v = args.w2v
        else:
            self.w2v = args.w2v = W2V(args)

        if hasattr(args, 'w_freq'):
            self.w_freq = args.w_freq
        else:
            self.w_freq = args.w_freq = WFreq(args)

        if hasattr(args, 'ent_name_id'):
            self.ent_name_id = args.ent_name_id
        else:
            self.ent_name_id = args.ent_name_id = EntNameID(args)

        if hasattr(args, 'e_freq_index'):
            self.e_freq_index = args.e_freq_index
        else:
            self.e_freq_index = args.e_freq_index = EFreqIndex(args)

        # **YD** deep-ed doesn't require to retrain the word embedding (from w2v) or
        # entity embedding (from previous entity embedding generation)
        # Future model may require to train them if necessary, it will be loaded in model instead of data loader.

        self.ent_lookup_table = self.load_ent_embed()
        self.word_lookup_table = self.w2v.M

    def load_ent_embed(self):
        weight = torch.load(self.ent_embed_file, map_location=torch.device('cpu'))
        weight[self.ent_name_id.unk_ent_thid] = 0

        look_up_table = nn.Embedding.from_pretrained(F.normalize(weight))

        print("look_up_table.weight.size(0)", look_up_table.weight.size(0))
        print("self.ent_name_id.get_total_num_ents()", self.ent_name_id.get_total_num_ents())
        assert look_up_table.weight.size(0) == self.ent_name_id.get_total_num_ents() + 1

        return look_up_table

    def empty_minibatch_with_ids(self, num_mentions):
        inputs = [
            [None, None],
            [None, None],
            None,
        ]

        # -- ctxt_w_ids
        # **YD** "unk_w_id" has been implemented
        inputs[0][0] = torch.ones(num_mentions, self.args.ctxt_window, dtype=torch.long) * self.w_freq.unk_w_id
        # -- TO BE FILLED : inputs[0][1] =  CTXT WORD VECTORS : num_mentions x args.ctxt_window x args.ent_vecs_size

        # -- e_wikiids
        # **YD** "args.num_cand_before_rerank" has been implemented
        inputs[1][0] = torch.ones(num_mentions, self.args.num_cand_before_rerank, dtype=torch.long) * \
                       self.ent_name_id.unk_ent_wikiid
        # -- TO BE FILLED : inputs[1][1] = ENT VECTORS: num_mentions x opt.num_cand_before_rerank x ent_vecs_size

        # -- p(e|m)
        inputs[2] = torch.zeros(num_mentions, self.args.num_cand_before_rerank)

        return inputs

    def process_one_line(self, line, minibatch_tensor, mb_index, for_training=False):
        parts = line.split('\t')

        # -- Ctxt word ids:
        # **YD** "parse_context" has been implemented
        ctxt_word_ids = self.parse_context(parts)
        minibatch_tensor[0][0][mb_index] = ctxt_word_ids

        # -- Entity candidates:
        # **YD** " parse_candidate_entities" has been implemented
        grd_trth_idx, grd_trth_ent_wikiid, ent_cand_wikiids, log_p_e_m = \
            self.parse_candidate_entities(parts, for_training, self.args.num_cand_before_rerank)

        # **YD** "get_ent_thids" has been implemented
        # print("ent_cand_wikiids", type(ent_cand_wikiids), ent_cand_wikiids)
        minibatch_tensor[1][0][mb_index] = self.ent_name_id.get_ent_thids(ent_cand_wikiids)
        # print("ent_cand_thids", minibatch_tensor[1][0][mb_index])

        # **YD** "get_wikiid_from_thid" has been implemented
        """
        print("grd_trth_idx", type(grd_trth_idx), grd_trth_idx)

        tmp = self.ent_name_id.get_wikiid_from_thid(int(minibatch_tensor[1][0][mb_index][grd_trth_idx]))
        print(
            "minibatch_tensor[1][0][mb_index][grd_trth_idx]",
            type(minibatch_tensor[1][0][mb_index][grd_trth_idx]),
            minibatch_tensor[1][0][mb_index][grd_trth_idx]
        )
        print("tmp", type(tmp), tmp)
        print("grd_trth_ent_wikiid", type(grd_trth_ent_wikiid), grd_trth_ent_wikiid)
        """

        assert (grd_trth_idx == -1 or
                self.ent_name_id.get_wikiid_from_thid(int(minibatch_tensor[1][0][mb_index][grd_trth_idx]))
                == grd_trth_ent_wikiid)

        # -- log p(e|m):
        minibatch_tensor[2][mb_index] = log_p_e_m

        return grd_trth_idx

    def parse_context(self, parts):
        ctxt_word_ids = torch.ones(self.args.ctxt_window, dtype=torch.long)
        lc = parts[3]
        lc_words = lc.split(' ')
        j = self.args.ctxt_window // 2 - 1
        i = len(lc_words) - 1

        # **YD** "get_id_from_word" has been implemented
        while j >= 0 and i >= 0:
            while i >= 1 and str(self.w_freq.get_id_from_word(lc_words[i])) == str(self.w_freq.unk_w_id):
                i -= 1
            # **YD** word_id data formate check
            ctxt_word_ids[j] = int(self.w_freq.get_id_from_word(lc_words[i]))
            j -= 1
            i -= 1

        rc = parts[4]
        rc_words = rc.split(' ')
        j = self.args.ctxt_window // 2
        i = 0

        # **YD** "get_id_from_word" has been implemented
        while j < self.args.ctxt_window and i < len(rc_words):
            while i < len(rc_words) - 1 and str(self.w_freq.get_id_from_word(rc_words[i])) == str(self.w_freq.unk_w_id):
                i += 1

            # **YD** word_id data format check
            ctxt_word_ids[j] = int(self.w_freq.get_id_from_word(rc_words[i]))

            i += 1
            j += 1

        # **YD** debug normal: print('ctxt_word_ids', ctxt_word_ids)
        return ctxt_word_ids

    # ---------------------- Entity candidates: ----------------
    def parse_num_cand_and_grd_trth(self, parts):
        assert parts[5] == 'CANDIDATES'
        if parts[6] == 'EMPTYCAND':
            return 0

        num_cand = 1
        while parts[6 + num_cand] != 'GT:':
            num_cand += 1

        return num_cand

    # --- Collect the grd truth label:
    # -- @return grd_trth_idx, grd_trth_ent_wikiid, grd_trth_prob
    def get_grd_trth(self, parts, num_cand, for_training):
        assert parts[6 + max(1, num_cand)] == 'GT:'
        grd_trth_str = parts[7 + max(1, num_cand)]
        grd_trth_parts = grd_trth_str.split(',')

        # **YD** data type is int, starting from 0
        grd_trth_idx = int(grd_trth_parts[0])
        if grd_trth_idx != -1:
            assert 0 <= grd_trth_idx < num_cand
            assert grd_trth_str == str(grd_trth_idx) + ',' + parts[6 + grd_trth_idx]
        else:
            assert not for_training

        grd_trth_prob = 0
        if grd_trth_idx >= 0:
            grd_trth_prob = min(1.0, max(1e-3, float(grd_trth_parts[2])))

        # **YD** "unk_ent_wikiid" has been implemented
        grd_trth_ent_wikiid = self.ent_name_id.unk_ent_wikiid

        if len(grd_trth_parts) >= 2:
            grd_trth_ent_wikiid = int(grd_trth_parts[1])

        return grd_trth_idx, grd_trth_ent_wikiid, grd_trth_prob

    # -- @return grd_trth_idx, grd_trth_ent_wikiid, ent_cand_wikiids, log_p_e_m, log_p_e
    # **YD** require heavily debugging
    def parse_candidate_entities(self, parts, for_training, orig_max_num_cand):

        num_cand = self.parse_num_cand_and_grd_trth(parts)
        grd_trth_idx, grd_trth_ent_wikiid, grd_trth_prob = self.get_grd_trth(parts, num_cand, for_training)
        if grd_trth_idx > orig_max_num_cand - 1:
            grd_trth_idx = -1

        # -- P(e|m) prior:
        log_p_e_m = torch.ones(orig_max_num_cand) * -1e8

        # -- Vector of entity candidates ids
        # **YD** "unk_ent_wikiid" has been implemented
        ent_cand_wikiids = torch.ones(orig_max_num_cand, dtype=torch.long) * self.ent_name_id.unk_ent_wikiid

        # -- Parse all candidates
        for cand_index in range(min(num_cand, orig_max_num_cand)):
            cand_parts = (parts[6 + cand_index]).split(',')

            cand_ent_wikiid = int(cand_parts[0])
            assert cand_ent_wikiid > 0

            ent_cand_wikiids[cand_index] = cand_ent_wikiid

            # **YD** "get_thid" has been implemented
            assert for_training or self.ent_name_id.get_thid(cand_ent_wikiid) != self.ent_name_id.unk_ent_thid
            # -- RLTD entities have valid id

            cand_p_e_m = min(1.0, max(1e-3, float(cand_parts[1])))

            log_p_e_m[cand_index] = math.log(cand_p_e_m)

            # **YD** "get_ent_freq" has been implemented
            # cand_p_e = self.e_freq_index.get_ent_freq(cand_ent_wikiid)

        # -- Reinsert grd truth for training only
        # **YD** seems like to put the last as the ground truth
        if grd_trth_idx == -1 and for_training:
            assert num_cand >= orig_max_num_cand
            grd_trth_idx = orig_max_num_cand - 1
            ent_cand_wikiids[grd_trth_idx] = grd_trth_ent_wikiid
            log_p_e_m[grd_trth_idx] = math.log(grd_trth_prob)

        # -- Sanity checks:
        #   assert(log_p_e_m[1] == torch.max(log_p_e_m), line)
        #   assert(log_p_e_m[2] == torch.max(log_p_e_m:narrow(1, 2, orig_max_num_cand - 1)), line)

        if grd_trth_idx != -1:
            assert grd_trth_ent_wikiid != self.ent_name_id.unk_ent_wikiid

            # print("ent_cand_wikiids[grd_trth_idx]", type(ent_cand_wikiids[grd_trth_idx]), ent_cand_wikiids[grd_trth_idx])
            # print("grd_trth_ent_wikiid", type(grd_trth_ent_wikiid), grd_trth_ent_wikiid)
            assert int(ent_cand_wikiids[grd_trth_idx]) == int(grd_trth_ent_wikiid)
            assert log_p_e_m[grd_trth_idx] == math.log(grd_trth_prob)
        else:
            assert not for_training

        return grd_trth_idx, grd_trth_ent_wikiid, ent_cand_wikiids, log_p_e_m

    def minibatch_to_correct_type(self, minibatch_tensor, targets, for_training):

        # -- ctxt_w_vecs : num_mentions x ctxt_window x ent_vecs_size
        # **YD** "word_lookup_table" has been implemented
        minibatch_tensor[0][1] = self.word_lookup_table(minibatch_tensor[0][0])

        # -- ent_vecs : num_mentions x max_num_cand x ent_vecs_size
        # **YD** "ent_lookup_table" has been implemented
        minibatch_tensor[1][1] = self.ent_lookup_table(minibatch_tensor[1][0])

        # -- log p(e|m) : num_mentions x max_num_cand
        # minibatch_tensor[3] = correct_type(minibatch_tensor[3])

        # **YD** "rerank" has been implemented
        # **YD** core debuggging!
        # print("before rerank!")
        # print("targets", targets)
        return self.rerank(minibatch_tensor, targets, for_training)

    def rerank(self, minibatch_tensor, targets, for_training):
        # **YD** "get_cand_ent_thids" has been implemented
        num_mentions = self.get_cand_ent_thids(minibatch_tensor).size(0)
        new_targets = torch.ones(num_mentions, dtype=torch.long) * (-1)

        # -- Average of word vectors in a window of (at most) size 50 around the mention.
        ctxt_vecs = minibatch_tensor[0][1]
        if self.args.ctxt_window > 50:
            # **YD** "narrow" has been implemented
            ctxt_vecs = ctxt_vecs.narrow(1, self.args.ctxt_window // 2 - 25, 50)

        # **YD** logic has been clear
        ctxt_vecs = ctxt_vecs.sum(1).view(num_mentions, self.args.ent_vecs_size)

        # **YD** "args.max_num_cand" has been implemented
        new_log_p_e_m = torch.ones(num_mentions, self.args.max_num_cand) * (-1e8)

        # **YD** "unk_ent_wikiid" has been implemented
        new_ent_cand_wikiids = torch.ones(
            num_mentions, self.args.max_num_cand, dtype=torch.long) * self.ent_name_id.unk_ent_wikiid

        new_ent_cand_vecs = torch.zeros(num_mentions, self.args.max_num_cand, self.args.ent_vecs_size)

        for k in range(num_mentions):
            ent_vecs = minibatch_tensor[1][1][k]
            scores = (ent_vecs * ctxt_vecs[k]).sum(1)
            #print("ent_vecs", type(ent_vecs), ent_vecs.size())
            #print("ctxt_vecs[k]", type(ctxt_vecs[k]), ctxt_vecs[k].size())
            assert scores.size(0) == self.args.num_cand_before_rerank, str(scores.size(0)) + ' ' + str(self.args.num_cand_before_rerank)

            # **YD** not sure the torch.sort usage
            _, ctxt_indices = torch.sort(scores, descending=True)

            added_indices = dict()
            for j in range(self.args.keep_e_ctxt):
                added_indices[ctxt_indices[j].item()] = 1

            j = 0
            while len(added_indices) < self.args.max_num_cand:
                added_indices[j] = 1
                j += 1

            new_grd_trth_idx = -1

            # print("added_indices", added_indices)
            for i, idx in enumerate(added_indices):
                new_ent_cand_wikiids[k][i] = minibatch_tensor[1][0][k][idx]
                new_ent_cand_vecs[k][i] = minibatch_tensor[1][1][k][idx]
                new_log_p_e_m[k][i] = minibatch_tensor[2][k][idx]

                if idx == targets[k]:
                    # **YD** "unk_ent_wikiid" has been implemented
                    #assert minibatch_tensor[1][0][k][idx] != self.ent_name_id.unk_ent_wikiid

                    if minibatch_tensor[1][0][k][idx] == self.ent_name_id.unk_ent_wikiid:
                        # print('k', k)
                        # print('idx', idx)
                        # print('minibatch_tensor[1][0][k]', minibatch_tensor[1][0][k])
                        raise ValueError('wrong!')
                    new_grd_trth_idx = i

            if targets[k] >= 0 and new_grd_trth_idx == -1:
                assert targets[k] >= self.args.keep_p_e_m

                # **YD** put the GT to first place.
                if for_training: # -- Reinsert the grd truth entity for training only
                    new_ent_cand_wikiids[k][0] = minibatch_tensor[1][0][k][targets[k]]
                    new_ent_cand_vecs[k][0] = minibatch_tensor[1][1][k][targets[k]]
                    new_log_p_e_m[k][0] = minibatch_tensor[2][k][targets[k]]
                    new_grd_trth_idx = 0

            new_targets[k] = new_grd_trth_idx

        minibatch_tensor[1][0] = new_ent_cand_wikiids
        minibatch_tensor[1][1] = new_ent_cand_vecs
        minibatch_tensor[2] = new_log_p_e_m

        return minibatch_tensor, new_targets

    # **YD** support function
    def get_cand_ent_wikiids(self, minibatch_tensor):
        cand_ent_thids = self.get_cand_ent_thids(minibatch_tensor)
        assert type(cand_ent_thids) is torch.Tensor
        num_mentions = cand_ent_thids.size(0)
        max_num_cand = cand_ent_thids.size(1)
        assert max_num_cand == self.args.max_num_cand

        t = torch.zeros(num_mentions, max_num_cand, dtype=torch.long)
        for i in range(num_mentions):
            for j in range(max_num_cand):
                # **YD** core bug,
                t[i][j] = self.ent_name_id.get_wikiid_from_thid(cand_ent_thids[i][j].item())
        return t

    def get_log_p_e_m(self, minibatch_tensor):
        return minibatch_tensor[2]

    def get_ctxt_word_ids(self, minibatch_tensor):
        return minibatch_tensor[0][0]

    def get_cand_ent_thids(self, minibatch_tensor):
        return minibatch_tensor[1][0]



