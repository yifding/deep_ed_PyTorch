import os
# import json
import argparse

from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID


class EFreqGen(object):
    def __init__(self, args, filter=10):
        self.args = args

        # filter is used to filter out entities have less frequencies
        self.filter = filter
        self.wiki = 'generated/crosswikis_wikipedia_p_e_m.txt'
        self.wiki_file = os.path.join(args.root_data_dir, self.wiki)
        assert os.path.isfile(self.wiki_file)

        self.out = 'generated/ent_wiki_freq.txt'
        self.out_file = os.path.join(args.root_data_dir, self.out)

        self.ent_name_id = EntNameID(self.args)

        self.entity_freqs = dict()
        self.read_from_txt()
        self.write_freqs()

    @property
    def dict(self):
        return self.entity_freqs

    def read_from_txt(self):

        num_lines = 0

        print('\nComputing wiki_e_frequency')
        with open(self.wiki_file, 'r') as reader:
            for line in reader:
                line = line.rstrip()
                num_lines += 1
                if num_lines % 2000000 == 0:
                    print('Processed ' + str(num_lines) + ' lines. ')

                parts = line.split('\t')
                num_parts = len(parts)

                for i in range(2, num_parts):
                    ent_str = parts[i].split(',')
                    ent_wikiid = int(ent_str[0])
                    freq = int(ent_str[1])

                    assert ent_wikiid >= 1
                    assert freq >= 1

                    if ent_wikiid not in self.dict:
                        self.dict[ent_wikiid] = freq
                    else:
                        self.dict[ent_wikiid] += freq

    def write_freqs(self):
        ent_freq = dict()
        for ent_wikiid, freq in self.dict.items():
            if freq >= self.filter:
                ent_freq[ent_wikiid] = freq

        sorted_ent_freq = sorted(ent_freq.items(), key=lambda x: x[1], reverse=True)

        total_freq = 0
        print('====> Now sorting and writing freqs_from_wiki.. \n')

        with open(self.out_file, 'w') as writer:
            for ent_wikiid, freq in sorted_ent_freq:
                s = str(ent_wikiid) + '\t' + self.ent_name_id.get_ent_name_from_wikiid(ent_wikiid) + '\t' + str(freq) + '\n'
                writer.write(s)
                total_freq += freq

        print('Total freq = ' + str(total_freq) + '\n')


def test(args):
    EFreqGen(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate the coverage percent of entity dictionary on evaluate GT entity'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )
    args = parser.parse_args()
    test(args)