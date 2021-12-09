import os
import math
import random
import bisect
import argparse

from deep_ed_PyTorch.words import LoadWFreqAandVecs

import torch


class WFreq(object):
    """
    main class to record word dictionary and its frequencies and samplings
    """
    def __init__(self, args):

        self.args = args
        self.w_freq = 'generated/word_wiki_freq.txt'
        self.w_freq_file = os.path.join(self.args.root_data_dir, self.w_freq)

        self.load_w_freq_and_vecs = LoadWFreqAandVecs(args)

        self.id2word = dict()
        self.word2id = dict()
        self.w_f_start = dict()
        self.w_f_end = dict()
        self.total_freq = 0.0

        self.w_f_at_unig_power_start = dict()
        self.w_f_at_unig_power_end = dict()
        self.total_freq_at_unig_power = 0.0

        self.total_num_words = 0

        # ATTENTION:
        # **YD** id starting from 1, may cause confusion in the embedding look ups.
        # should be consistent with entity embedding look ups.

        self.unk_w_id = 1
        self.word2id['UNK_W'] = self.unk_w_id
        self.id2word[self.unk_w_id] = 'UNK_W'

        self.pad_w_id = 0
        self.word2id['PAD_W'] = self.pad_w_id
        self.id2word[self.pad_w_id] = 'PAD_W'

        self.load_w_freq()

        # **YD** extra self.list for faster random process
        self.word_freq_select = list(self.w_f_start.values()) + [self.total_freq]
        self.word_freq_unig_select = list(self.w_f_at_unig_power_start.values()) + [self.total_freq_at_unig_power]

        """
        #tiny example of the random logic:
        word2id = {
            'UNK_W': 1,
            'a': 2,
            'b': 3,
            'c': 4,
        }
        
        id2word = {
            1:'UNK_W',
            2:'a',
            3:'b',
            4:'c',
        }
        
        w_f_start = {
            2:0,
            3:1,
            4:3,
        }
        
        total_freq = 7
        
        word_freq_select = list(w_f_start.values()) + [total_freq]
        assert word_freq_select == [0, 1, 3, 7]
        #k = random.uniform(low=1, high=word_freq_select[-1])
        
        def rand_id(k, total_freq, word_freq_select):
            return bisect.bisect_left(word_freq_select, k) + 1
        
        assert rand_id(1, total_freq, word_freq_select) == 2
        assert rand_id(2, total_freq, word_freq_select) == 3
        assert rand_id(3, total_freq, word_freq_select) == 3
        assert rand_id(4, total_freq, word_freq_select) == 4
        assert rand_id(5, total_freq, word_freq_select) == 4
        assert rand_id(6, total_freq, word_freq_select) == 4
        assert rand_id(7, total_freq, word_freq_select) == 4
        """

        if hasattr(args, 'unit_tests'):
            print(self.get_w_id_unigram(self.get_id_from_word('the')))
            print(self.get_w_id_unigram(self.get_id_from_word('of')))
            print(self.get_w_id_unigram(self.get_id_from_word('and')))
            print(self.get_w_id_unigram(self.get_id_from_word('romania')))

    def load_w_freq(self):
        tmp_wid = 1
        with open(self.w_freq_file, 'r') as reader:
            for line in reader:
                line = line.rstrip()
                parts = line.split('\t')
                w = parts[0]
                if w in self.load_w_freq_and_vecs.dict:
                    tmp_wid += 1
                    w_id = tmp_wid
                    self.id2word[w_id] = w
                    self.word2id[w] = w_id

                    w_f = int(parts[1])
                    assert w_f > 0
                    if w_f < 100:
                        w_f = 100

                    self.w_f_start[w_id] = self.total_freq
                    self.total_freq += w_f
                    self.w_f_end[w_id] = self.total_freq

                    self.w_f_at_unig_power_start[w_id] = self.total_freq_at_unig_power
                    self.total_freq_at_unig_power += w_f**self.args.unig_power
                    self.w_f_at_unig_power_end[w_id] = self.total_freq_at_unig_power

        self.total_num_words = tmp_wid
        print('    Done loading word freq index. Num words = ' +
              str(self.total_num_words) + '; total freq = ' + str(self.total_freq))

    # --------------------------------------------
    # supporting function s
    # --------------------------------------------

    '''
    def total_num_words(self):
        return self.total_num_words
    '''

    def contains_w_id(self, w_id):
        w_id = int(w_id)
        assert 1 <= w_id <= self.total_num_words, w_id
        return w_id != self.unk_w_id

    def get_word_from_id(self, w_id):
        w_id = int(w_id)
        assert 0 <= w_id <= self.total_num_words
        return self.id2word[w_id]

    def get_id_from_word(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.unk_w_id

    def contains_w(self, w):
        return self.contains_w_id(self.get_id_from_word(w))

    """ **YD** original has + 1 in the code, may not be right
    # deep-ed/words/w_freq/w_freq_index.lua#L85
    -- word frequency:
    function get_w_id_freq(w_id)
      assert(contains_w_id(w_id), w_id)
      return w_freq.w_f_end[w_id] - w_freq.w_f_start[w_id] + 1
    end
    
    """
    # -- word frequency:
    def get_w_id_freq(self, w_id):
        assert self.contains_w_id(w_id)
        return self.w_f_end[w_id] - self.w_f_start[w_id]

    # -- p(w) prior:
    def get_w_id_unigram(self, w_id):
        return self.get_w_id_freq(w_id) / self.total_freq

    # **YD** may change to apply in torch.Tensor to run faster
    def get_w_tensor_log_unigram(self, vec_w_ids):
        assert type(vec_w_ids) is torch.Tensor
        assert len(vec_w_ids.shape) == 2
        v = torch.zeros(vec_w_ids.shape)

        for i in range(vec_w_ids.size(0)):
            for j in range(vec_w_ids.size(1)):
                v[i][j] = math.log(self.get_w_id_unigram(vec_w_ids[i][j]))

        return v

    """ **YD** original random word sampling function
    -- Generates an random word sampled from the word unigram frequency.
    local function random_w_id(total_freq, w_f_start, w_f_end)
      local j = math.random() * total_freq
      local i_start = 2
      local i_end = total_num_words()
    
      while i_start <= i_end do
        local i_mid = math.floor((i_start + i_end) / 2)
        local w_id_mid = i_mid
        if w_f_start[w_id_mid] <= j and j <= w_f_end[w_id_mid] then
          return w_id_mid
        elseif (w_f_start[w_id_mid] > j) then
          i_end = i_mid - 1
        elseif (w_f_end[w_id_mid] < j) then
          i_start = i_mid + 1
        end
      end
      print(red('Binary search error !!'))
    end
    """

    # -- Generates an random word sampled from the word frequency.
    def random_w_id(self):
        k = random.randint(self.word_freq_select[1], self.word_freq_select[-1])
        output = bisect.bisect_left(self.word_freq_select, k) + 1
        assert 2 <= output <= self.total_num_words
        return output

    # -- Generates an random word sampled from the word frequency with power unigram.
    def random_unigram_at_unig_power_w_id(self):
        k = random.randint(1, math.floor(self.word_freq_unig_select[-1]))
        output = bisect.bisect_left(self.word_freq_unig_select, k) + 1
        assert 2 <= output <= self.total_num_words
        return output

    def unit_test_random_unigram_at_unig_power_w_id(self, k_samples):
        """
        **YD**
        'In' and 'The' appears too high, may cause the negative sampling not work
        """
        empirical_dist = dict()
        for i in range(k_samples):
            w_id = self.random_unigram_at_unig_power_w_id()
            assert w_id != self.unk_w_id
            empirical_dist[w_id] = 1 if w_id not in empirical_dist else empirical_dist[w_id] + 1

        print('Now sorting ..')
        empirical_dist = dict(sorted(empirical_dist.items(), key=lambda x: x[1], reverse=True))
        s = ''
        for i, key in enumerate(empirical_dist.keys()):
            if i >= 100:
                break
            s += self.get_word_from_id(key) + '{' + str(empirical_dist[key]) + '};'

        print('Unit test random sampling: ' + s)

    def get_w_unnorm_unigram_at_power(self, w_id):
        return self.get_w_id_unigram(w_id)**self.args.unig_power


def test(args):
    w_freq = WFreq(args)
    w_freq.unit_test_random_unigram_at_unig_power_w_id(100000)


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


    args = parser.parse_args()
    test(args)