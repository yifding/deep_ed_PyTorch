import os
import argparse

from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID


class MergeCrosswikisWiki(object):
    def __init__(self, args):
        self.args = args
        assert os.path.isdir(args.root_data_dir)

        self.wiki = 'generated/wikipedia_p_e_m.txt'
        self.wiki_file = os.path.join(args.root_data_dir, self.wiki)
        self.cross_wiki = 'basic_data/p_e_m_data/crosswikis_p_e_m.txt'
        self.cross_wiki_file = os.path.join(args.root_data_dir, self.cross_wiki)
        self.out = 'generated/crosswikis_wikipedia_p_e_m.txt'
        self.out_file = os.path.join(args.root_data_dir, self.out)

        self.ent_name_id = EntNameID(self.args)

        self.merged_e_m_counts = dict()
        self.load_from_wiki()
        self.merged_with_cross_wiki()
        self.write_p_e_m()

    @property
    def dict(self):
        return self.merged_e_m_counts

    def load_from_wiki(self):

        print('\nMerging Wikipedia and Crosswikis p_e_m')
        print('Process Wikipedia')

        self.load_from_txt(self.wiki_file)

    def merged_with_cross_wiki(self):
        print('Process Crosswikis')

        self.load_from_txt(self.cross_wiki_file)

    def load_from_txt(self, txt_file):
        with open(txt_file) as reader:
            for line in reader:
                line = line.rstrip('\t\n')
                parts = line.split('\t')
                mention = parts[0]

                if 'Wikipedia' not in mention and 'wikipedia' not in mention:
                    if mention not in self.dict:
                        self.dict[mention] = dict()

                    assert int(parts[1]) >= 1, "wiki total frequency error: {}, {}".format(repr(line), parts[1])
                    total_freq = int(parts[1])

                    num_ents = len(parts) - 2
                    for i in range(2, num_ents + 2):
                        ent_str = parts[i].split(',')

                        ent_wikiid = int(ent_str[0])
                        try:
                            ent_wikiid >= 1, "wiki id error: {}, {}".format(repr(line), ent_wikiid)

                        except:
                            print("wiki id error: {} :**: {}".format(repr(line), ent_wikiid))
                            raise 1

                        assert int(ent_str[1]) >= 1, "wiki frequency error: {}, {}".format(repr(line), ent_str[1])
                        freq = int(ent_str[1])

                        if ent_wikiid not in self.dict[mention]:
                            self.dict[mention][ent_wikiid] = 0

                        self.dict[mention][ent_wikiid] += freq

    def write_p_e_m(self):
        print('====> Now sorting and writing p_m_e_from_wiki..')
        with open(self.out_file, 'w') as writer:
            for mention in self.dict:
                el_list = self.dict[mention]
                sorted_dict_items = sorted(el_list.items(), key=lambda x: x[1], reverse=True)
                s = ''
                total_freq = 0
                num_ents = 0
                for wikiid, freq in sorted_dict_items:
                    if self.ent_name_id.is_valid_ent(wikiid):
                        s += str(wikiid) + ',' + str(freq) + ','
                        s += self.ent_name_id.get_ent_name_from_wikiid(wikiid).replace(' ', '_') + '\t'
                        total_freq += freq
                        num_ents += 1

                        if num_ents >= 100:
                            break

                s = mention + '\t' + str(total_freq) + '\t' + s + '\n'
                writer.write(s)

        print('    Done sorting and writing.')


def test(args):
    merge_crosswikis_wiki = MergeCrosswikisWiki(args)
    return merge_crosswikis_wiki


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
    merge_crosswikis_wiki = test(args)
