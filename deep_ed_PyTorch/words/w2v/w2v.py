import os
import copy
import argparse

import deep_ed_PyTorch.utils as utils
from deep_ed_PyTorch.words import LoadWFreqAandVecs
from deep_ed_PyTorch.words.w_freq import WFreq

import torch
import torch.nn as nn
import torch.nn.functional as F


class W2V(object):
    def __init__(self, args):
        """
        self.w_freq = WFreq(args)
        self.load_w_freq_and_vecs = self.w_freq.load_w_freq_and_vecs
        """

        if hasattr(args, 'w_freq'):
            self.w_freq = args.w_freq
        else:
            self.w_freq = args.w_freq = WFreq(args)
        self.load_w_freq_and_vecs = self.w_freq.load_w_freq_and_vecs

        self.word_vecs_size = 300
        assert args.word_vecs == 'w2v'

        self.w2v_model = self.load_w_freq_and_vecs.model
        self.w2v_torch_file = os.path.join(args.root_data_dir, 'generated/GoogleNews-vectors-negative300.pt')

        self.M = torch.zeros(self.w_freq.total_num_words + 1, self.word_vecs_size)
        self.load_w2v_weight_subset()

        """
        -- Move the word embedding matrix on the GPU if we do some training. 
        -- In this way we can perform word embedding lookup much faster.
        if opt and string.find(opt.type, 'cuda') then
          w2vutils.M = w2vutils.M:cuda()
        end
        """

    # ---------- Define additional functions -----------------
    # -- word -> vec

    def get_w_vec(self, word):
        w_id = self.w_freq.get_id_from_word(word)
        return self.M(torch.tensor(w_id)).clone()

    # -- word_id -> vec
    def get_w_vec_from_id(self, w_id):
        return self.M(torch.tensor(w_id)).clone()

    # -- lookup_w_vecs
    def lookup_w_vecs(self, word_id_tensor):

        # print(type(word_id_tensor))
        # print(word_id_tensor.dtype)
        # print(word_id_tensor.dim())
        assert type(word_id_tensor) is torch.Tensor and word_id_tensor.dtype is torch.long \
               and (word_id_tensor.dim() == 1 or word_id_tensor.dim() == 2)
        return self.M(word_id_tensor)

    # -- Phrase embedding using average of vectors of words in the phrase
    def phrase_avg_vec(self, phrase):
        words = utils.split_in_words(phrase)
        num_existent_words = 0
        vec = torch.zeros(self.word_vecs_size)
        for w in words:
            w_id = self.w_freq.get_id_from_word(w)
            if w_id != self.w_freq.unk_w_id:
                vec += self.get_w_vec_from_id(w_id)
        if num_existent_words > 0:
            vec /= num_existent_words
        return vec

    def top_k_closest_words(self, vec, k=1):
        distances = torch.mv(self.M.weight, vec)

        # **YD** topk not implemented
        best_scores, best_word_ids = torch.topk(distances, k)
        return_words = [self.w_freq.get_word_from_id(int(i)) for i in best_word_ids]
        return_distances = [distances[i] for i in best_word_ids]

        return return_words, return_distances

    def most_similar2word(self, word, k=1):
        v = self.get_w_vec(word)
        neighbors, scores = self.top_k_closest_words(v, k)
        s = 'To word ' + word + ' : '
        for neighbor, score in zip(neighbors, scores):
            s += ' ' + neighbor + ' ' + '{:.3f}'.format(score) + '   '
        print(s)

    def most_similar2vec(self, v, k=1):
        neighbors, scores = self.top_k_closest_words(v, k)
        s = ''
        for neighbor, score in zip(neighbors, scores):
            s += ' ' + neighbor + ' ' + '{:.3f}'.format(score) + '   '
        print(s)

    def load_w2v_weight_subset(self):
        if os.path.isfile(self.w2v_torch_file):
            print('  ---> from pt file.')
            self.M = torch.load(self.w2v_torch_file)
        else:
            print('  ---> pt file NOT found. Building w2v from the complete w2v matrix instead (slower).')

            unk_w_id = self.w_freq.unk_w_id

            self.M = torch.zeros(self.w_freq.total_num_words + 1, self.word_vecs_size)

            num_phrases = 0
            for w in self.w2v_model.vocab:
                w_id = self.w_freq.get_id_from_word(w)
                if w_id != unk_w_id:
                    self.M[w_id] = torch.from_numpy(self.w2v_model.get_vector(w).copy())

            print('Num words = ' + str(self.w_freq.total_num_words) + '. Num phrases = ' + str(num_phrases))
            torch.save(self.M, self.w2v_torch_file)

        # **YD** normalize and set unk_w_id to 0
        self.M[self.w_freq.unk_w_id] = 0
        self.M = nn.Embedding.from_pretrained(F.normalize(self.M))


def test(args):
    w2v = W2V(args)
    w2v.most_similar2word('nice', 5)
    w2v.most_similar2word('france', 5)
    w2v.most_similar2word('hello', 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='load word frequency and vectors',
        allow_abbrev=False,
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
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

    args = parser.parse_args()
    test(args)