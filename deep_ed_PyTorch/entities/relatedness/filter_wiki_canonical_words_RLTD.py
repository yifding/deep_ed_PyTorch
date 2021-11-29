import os
import argparse

from deep_ed_PyTorch.entities.relatedness import REWTR


class FilterWikiCanonicalWordsRLTD(object):
    def __init__(self, args):
        self.args = args

        self.input = 'generated/wiki_canonical_words.txt'
        self.input_file = os.path.join(args.root_data_dir, self.input)

        self.output = 'generated/wiki_canonical_words_RLTD.txt'
        self.output_file = os.path.join(args.root_data_dir, self.output)

        # **YD** self.rewtr not implemented
        self.rewtr = REWTR(args)

        self.read_and_write()

    def read_and_write(self):
        print('\nStarting dataset filtering.')
        num_lines = 0
        ouf = open(self.output_file, 'w')
        with open(self.input_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\t\n')
                num_lines += 1
                if num_lines % 500000 == 0:
                    print('    =======> processed ' + str(num_lines) + ' lines')

                parts = line.split('\t')
                assert len(parts) == 3, 'length error of input wiki_canonical_words.txt'

                ent_wikiid = int(parts[0])
                ent_name = parts[1]

                assert ent_wikiid > 0, 'invalid entity wikiid'

                if ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid:
                    ouf.write(line + '\n')

        ouf.close()


def test(args):
    FilterWikiCanonicalWordsRLTD(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate wiki canonical words rltd'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    print(args)
    test(args)