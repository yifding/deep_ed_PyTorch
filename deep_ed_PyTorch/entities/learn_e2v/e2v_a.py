# **YD** find most similar word to certain entity,
# using cosine similarity between its entity embedding with word embedding

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.words.w2v import W2V
from deep_ed_PyTorch.entities.EX_wiki_words import ExWikiWords
from deep_ed_PyTorch.entities.ent_name2id_freq.ent_name_id import EntNameID

from deep_ed_PyTorch.words import StopWords
from model_a import ModelA
# import WFreq; WFreq.contains_w


class E2V(object):
    def __init__(self, args):
        self.args = args

        self.stop_words = StopWords()

        """
        self.w2v = W2V(self.args)
        self.ex_wiki_words = ExWikiWords()
        self.ent_name_id = EntNameID(self.args)
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

        # **YD** initial parameters for "invalid_ent_wikiids_stats"
        self.num_invalid_ent_wikiids = 0
        self.total_ent_wiki_vec_requests = 0
        self.last_wrote = 0

    def invalid_ent_wikiids_stats(self, ent_thid):
        self.total_ent_wiki_vec_requests += 1

        if ent_thid == self.ent_name_id.unk_ent_thid:
            self.num_invalid_ent_wikiids += 1

        if self.num_invalid_ent_wikiids % 15000 == 0 and self.num_invalid_ent_wikiids != self.last_wrote:
            self.last_wrote = self.num_invalid_ent_wikiids
            perc = self.num_invalid_ent_wikiids / self.total_ent_wiki_vec_requests
            print('Perc invalid ent wikiids = ' + str(perc) + ' . Absolute num = ' + str(self.num_invalid_ent_wikiids))

    # -- ent id -> vec
    # **YD** different from intial implementation, requires extra "model_a" as input to obtain entity embedding.
    def geom_entwikiid2vec(self, ent_wikiid, model_a):
        ent_thid = self.ent_name_id.get_thid(ent_wikiid)
        self.invalid_ent_wikiids_stats(ent_thid)

        index = torch.tensor(ent_thid)

        """
        if hasattr(self.args, 'type') and 'cuda' in self.args.type:
            index = index.cuda()

        else:
            if hasattr(self.args, 'type'):
                print('type Yes')
            else:
                print('type No')

            if 'cuda' in 'self.cuda':
                print('cuda Yes')
            else:
                print('cuda No')
        """

        ent_vec = F.normalize(model_a.entity_embedding(index), dim=-1)
        return ent_vec

    # -- ent name -> vec
    def geom_entname2vec(self, ent_name, model_a):
        return self.geom_entwikiid2vec(self.ent_name_id.get_ent_wikiid_from_name(ent_name), model_a)

    def entity_similarity(self, e1_wikiid, e2_wikiid, model_a):
        e1_vec = self.geom_entwikiid2vec(e1_wikiid, model_a)
        e2_vec = self.geom_entwikiid2vec(e2_wikiid, model_a)

        # **YD** the output may not be correct, it outputs a 1D vector rather than a float number as i expected
        return e1_vec.dot(e2_vec)

    def geom_top_k_closest_words(self, ent_name, ent_vec, k=1):
        tf_map = self.ex_wiki_words.ent_wiki_words_4EX[ent_name]
        w_not_found = dict()

        for w in tf_map:
            if tf_map[w] >= 10:
                w_not_found[w] = tf_map[w]

        # **YD** may not be right
        distances = torch.mv(self.w2v.M.weight, ent_vec)

        best_scores, best_word_ids = torch.topk(distances, k)

        return_words = []
        return_scores = []
        for id in best_word_ids:
            w = self.w2v.w_freq.get_word_from_id(id)
            if self.stop_words.is_stop_word_or_number(w):
                return_words.append('STOP_OR_NUMBER: ' + w)
            elif w in tf_map:
                if tf_map[w] >= 15:
                    return_words.append('FREQUENT: ' + w + '{' + str(tf_map[w]) + '}')
                else:
                    return_words.append('RARE: ' + w + '{' + str(tf_map[w]) + '}')
                if w in w_not_found:
                    del w_not_found[w]
            else:
                return_words.append(w)
            return_scores.append(distances[id])

        return return_words, best_scores, w_not_found

    def geom_most_similar_words_to_ent(self, ent_name, model_a, k=1):
        # **YD** "get_ent_wikiid_from_name" has been implemented
        ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(ent_name)

        # **YD** "geom_entname2vec" has been implemented
        ent_vec = self.geom_entname2vec(ent_name, model_a)

        print('\nTo entity: ' + ent_name + '; vec norm = ' + str(ent_vec.norm()) + ':')

        # **YD** "geom_top_k_closest_words" has been implemented
        neighbors, scores, w_not_found = self.geom_top_k_closest_words(ent_name, ent_vec, k)

        s = 'WORDS MODEL: '
        for neighbor, score in zip(neighbors, scores):
            s += ' ' + str(neighbor) + '-' + '{:.3f}'.format(score) + '  '
        print(s)

        # **YD** print w_not_found not implemented
        st = 'WORDS NOT FOUND: '
        for w, tf in w_not_found.items():
          st += w + '{' + str(tf) + '};'
        print('\n' + st)
        print('============================================================================')

    def geom_unit_tests(self, model_a):
        """
        Function to perform test on small pre-selected entities 'ExWikiWords'
        """
        print('\nWords to Entity Similarity test:')
        for entity in self.ex_wiki_words.ent_names_4EX:
            # **YD** "geom_most_similar_words_to_ent" not implemented
            self.geom_most_similar_words_to_ent(entity, model_a, 200)


def test(args):
    e2v = E2V(args)
    model_a = ModelA(args)
    e2v.geom_unit_tests(model_a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='load word frequency and vectors',
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
        '--word_vecs',
        type=str,
        default='w2v',
        help='word embedding to use',
    )

    parser.add_argument(
        '--unig_power',
        type=float,
        default=0.6,
        help='unigram power to sample words',
    )

    parser.add_argument(
        '--entities',
        type=str,
        default='RLTD',
        choices=['RLTD', '4EX', 'ALL'],
        help='Set of entities for which we train embeddings: 4EX (tiny, for debug) |'
             ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
    )

    args = parser.parse_args()
    args.ent_vecs_size = 300
    args.init_vecs_title_words = True
    test(args)



