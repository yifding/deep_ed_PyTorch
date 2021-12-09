import os
import json
import html
import argparse

from deep_ed_PyTorch.data_gen.gen_p_e_m import UnicodeMap
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID


class GenPEMFromYago(object):
    def __init__(self, args):
        self.args = args
        self.yago = 'basic_data/p_e_m_data/aida_means.tsv'
        self.yago_file = os.path.join(self.args.root_data_dir, self.yago)
        assert os.path.isfile(self.yago_file)

        self.out = 'generated/yago_p_e_m.txt'
        self.out_file = os.path.join(self.args.root_data_dir, self.out)

        self.ent_name_id = EntNameID(self.args)
        self.unicode_map = UnicodeMap()

        self.wiki_e_m_counts = dict()
        self.load_from_yago()
        self.write_p_e_m()

    @property
    def dict(self):
        return self.wiki_e_m_counts

    def load_from_yago(self):
        num_lines = 0
        num_mention = 0

        print('\nComputing YAGO p_e_m')

        with open(self.yago_file, 'r') as reader:
            for line in reader:
                line = line.rstrip()
                num_lines += 1
                if num_lines % 5000000 == 0:
                    print('Processed ' + str(num_lines) + ' lines.')

                parts = line.split('\t')
                assert len(parts) == 2
                assert parts[0][0] == parts[0][-1] == '"'

                mention = parts[0][1:-1]
                ent_name = parts[1]
                # ent_name = ent_name.replace('&amp;', '&')
                # ent_name = ent_name.replace('&quot;', '"')
                # **YD** use html.unescape to process all the symbols from html to original symbol string
                ent_name = html.unescape(ent_name)
                while '\\u' in ent_name:
                    start = ent_name.find('\\u')
                    code = ent_name[start: start + 6]
                    replace = self.unicode_map.dict[code]

                    ent_name = ent_name.replace(code, replace)

                ent_name = self.ent_name_id.preprocess_ent_name(ent_name)
                ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(ent_name, True)
                if ent_wikiid != self.ent_name_id.unk_ent_wikiid:
                    num_mention += 1
                    if mention not in self.dict:
                        self.dict[mention] = dict()

                    # **YD** yago dictionary only reads 1, may be questionable
                    self.dict[mention][ent_wikiid] = 1

        print('Computing YAGO p_e_m done, process {} mentions!'.format(num_mention))

    def write_p_e_m(self):
        print('====> Now sorting and writing p_m_e_from_wiki..')
        with open(self.out_file, 'w') as writer:
            for mention in self.wiki_e_m_counts:
                el_list = self.wiki_e_m_counts[mention]
                sorted_dict_items = sorted(el_list.items(), key=lambda x: x[1], reverse=True)
                s = ''
                total_freq = 0
                for wikiid, freq in sorted_dict_items:
                    s += str(wikiid) + ','
                    s += self.ent_name_id.get_ent_name_from_wikiid(wikiid).replace(' ', '_') + '\t'
                    total_freq += 1

                s = mention + '\t' + str(total_freq) + '\t' + s + '\n'
                writer.write(s)

        print('    Done sorting and writing.')


def test(args):
    gen_p_e_m_from_yago = GenPEMFromYago(args)
    return gen_p_e_m_from_yago


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate the prior dictionary p(e|m) from wikipedia'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    gen_p_e_m_from_yago = test(args)
