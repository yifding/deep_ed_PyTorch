import os
import torch
import argparse

import gensim

from deep_ed_PyTorch.words import StopWords


class LoadWFreqAandVecs(object):
    """
    Main file to load 'word2vec' vectors, vocabulary and index2word, generate common word dictionary

    """
    def __init__(self, args):
        self.args = args
        self.default_path = os.path.join(args.root_data_dir, 'basic_data/wordEmbeddings/')

        self.w2v_path = os.path.join(self.default_path, 'Word2Vec/' + 'GoogleNews-vectors-negative300.bin')
        print('====> load w2v model from file')
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
        print('    ----> load w2v model from file finished')
        #self.w2v_vector = self.model.vectors
        #self.index2word = self.model.index2word
        #self.vocab = self.model.vocab

        self.input = 'generated/word_wiki_freq.txt'
        self.input_file = os.path.join(self.args.root_data_dir, self.input)
        self.output = 'generated/common_top_words_freq_vectors_' + self.args.word_vecs + '.pt'
        self.output_file = os.path.join(self.args.root_data_dir, self.output)

        self.stop_words = StopWords()

        self.common_w2v_freq_words = None
        self.generate()

    def generate(self):
        if os.path.isfile(self.output_file):
            print('  ---> from pt file.')
            self.common_w2v_freq_words = torch.load(self.output_file)

            '''    
            self.common_w2v_freq_words = dict(
                (key, int(value)) for key, value in torch_output.items()
            )
            '''

        else:
            print('  ---> pt file NOT found. Loading from disk instead (slower). Out file = ' + self.output_file)
            print('   word freq index ...')
            freq_words = dict()
            num_freq_words = 1

            with open(self.input_file) as reader:
                for line in reader:
                    line = line.rstrip()
                    parts = line.split('\t')
                    # print(parts)
                    w = parts[0]
                    w_f = int(parts[1])
                    assert w_f > 0
                    # **YD** is_step_word_or_number has been implemented
                    if not self.stop_words.is_stop_word_or_number(w):
                        freq_words[w] = w_f
                        num_freq_words += 1

            print('   word vectors index ...')
            assert self.args.word_vecs == 'w2v'

            self.common_w2v_freq_words = dict()
            for w in self.model.vocab:
                if w in freq_words:
                    self.common_w2v_freq_words[w] = 1

            print('Writing pt File for future usage. Next time loading will be faster!')
            torch.save(self.common_w2v_freq_words, self.output_file)

    @property
    def dict(self):
        return self.common_w2v_freq_words


def test(args):
    LoadWFreqAandVecs(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='load word frequency and vectors',
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

    args = parser.parse_args()
    test(args)
