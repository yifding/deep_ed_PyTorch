import os
import argparse


class WFreqGen(object):
    def __init__(self, args):
        self.args = args
        self.wiki_word = 'generated/wiki_canonical_words.txt'
        self.wiki_word_file = os.path.join(args.root_data_dir, self.wiki_word)

        self.out = 'generated/word_wiki_freq.txt'
        self.out_file = os.path.join(args.root_data_dir, self.out)

        self.freq_limit = 10

        self.word_freqs = dict()
        self.read_and_write()

    def read_and_write(self):
        num_lines = 0

        with open(self.wiki_word_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\t\n')
                num_lines += 1
                if num_lines % 100000 == 0:
                    print('Processed ' + str(num_lines) + ' lines. ')

                parts = line.split('\t')
                words = parts[2].split(' ')

                for w in words:
                    if w not in self.word_freqs:
                        self.word_freqs[w] = 1
                    else:
                        self.word_freqs[w] += 1

        # -- Writing word frequencies
        print('Sorting and writing')
        filter_list = [(w, self.word_freqs[w]) for w in self.word_freqs if self.word_freqs[w]>= self.freq_limit]
        sort_filter_list = sorted(filter_list, key=lambda x: x[1], reverse=True)
        with open(self.out_file, 'w') as writer:
            total_freq = 0
            for w,f in sort_filter_list:
                writer.write(w + '\t' + str(f) + '\n')
                total_freq += f

        print('Total freq = ' + str(total_freq) + '\n')


def test(args):
    WFreqGen(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate word frequencies'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    print(args)
    test(args)