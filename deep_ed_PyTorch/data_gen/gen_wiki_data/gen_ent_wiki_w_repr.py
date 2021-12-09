import os
import argparse

from deep_ed_PyTorch import utils
from deep_ed_PyTorch.data_gen.parse_wiki_dump import ParseWikiDumpTool
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID, EFreqIndex


class GenEntWikiPage(object):
    def __init__(self, args):
        self.args = args

        self.wiki = 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt'
        self.wiki_file = os.path.join(args.root_data_dir, self.wiki)

        self.out = 'generated/wiki_canonical_words.txt'
        self.out_file = os.path.join(args.root_data_dir, self.out)

        self.empty = 'generated/empty_page_ents.txt'
        self.empty_file = os.path.join(args.root_data_dir, self.empty)

        self.parse_wiki_dump_tool = ParseWikiDumpTool(args)
        self.ent_name_id = self.parse_wiki_dump_tool.ent_name_id
        self.e_freq_index = EFreqIndex(args)

        self.load_and_write()

    def load_and_write(self):
        print('\nExtracting text only from Wiki dump. Output is wiki_canonical_words.txt '
              'containing on each line an Wiki entity with the list of all words in its canonical Wiki page.')

        num_lines = 0
        num_valid_ents = 0
        num_error_ents = 0 # Probably list or disambiguation pages.

        # **YD** get_map_all_valid_ents not implemented
        empty_valid_ents = self.ent_name_id.get_map_all_valid_ents()

        cur_words = ''
        cur_ent_wikiid = -1

        ouf = open(self.out_file, 'w')
        with open(self.wiki_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\n\t')

                num_lines += 1
                if num_lines % 5000000 == 0:
                    print('Processed ' +  str(num_lines) + ' lines. Num valid ents = ' +
                          str(num_valid_ents) + '. Num errs = ' + str(num_error_ents))

                if '<doc id="' not in line and '</doc>' not in line:
                    # **YD** extract_text_and_hyp not implemented
                    _, text, _, _, _, _ = self.parse_wiki_dump_tool.extract_text_and_hyp(line, False)
                    words = utils.split_in_words(text)
                    cur_words = cur_words + ' '.join(words) + ' '

                elif '<doc id="' in line:
                    if cur_ent_wikiid > 0 and cur_words != '':
                        if cur_ent_wikiid != self.ent_name_id.unk_ent_wikiid \
                                and self.ent_name_id.is_valid_ent(cur_ent_wikiid):

                            # **YD** get_ent_name_from_wikiid not implemented
                            ouf.write(str(cur_ent_wikiid) + '\t' +
                                      self.ent_name_id.get_ent_name_from_wikiid(cur_ent_wikiid) +
                                      '\t' + cur_words + '\n')

                            # **YD** delete dictionary, not sure the logic here
                            if cur_ent_wikiid in empty_valid_ents:
                                del empty_valid_ents[cur_ent_wikiid]
                            # empty_valid_ents[cur_ent_wikiid] = nil

                            num_valid_ents += 1
                        else:
                            num_error_ents += 1

                    cur_ent_wikiid = self.parse_wiki_dump_tool.extract_page_entity_title(line)

                    assert cur_ent_wikiid >= 1
                    cur_words = ''
        ouf.close()

        # -- Num valid ents = 4126137. Num errs = 332944
        print('    Done extracting text only from Wiki dump. Num valid ents = ' +
              str(num_valid_ents) + '. Num errs = ' + str(num_error_ents))

        print('Create file with all entities with empty Wikipedia pages.')

        empty_ents = dict((ent_wikiid, self.e_freq_index.get_ent_freq(ent_wikiid))
                          for ent_wikiid in empty_valid_ents)
        sort_ents = dict(sorted(empty_ents.items(), key=lambda x: x[1], reverse=True))

        with open(self.empty_file, 'w') as writer:
            for ent_wikiid in sort_ents:
                writer.write(str(ent_wikiid) + '\t' + self.ent_name_id.get_ent_name_from_wikiid(ent_wikiid)
                             + '\t' + str(sort_ents[ent_wikiid]) + '\n')
        print('    Done')


def test(args):
    GenEntWikiPage(args)


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
    print(args)
    test(args)