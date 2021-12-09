import os
import json
import argparse

from collections import defaultdict

from deep_ed_PyTorch.data_gen.parse_wiki_dump import ParseWikiDumpTool

# 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt'


class GenPEMFromWiki(object):
    def __init__(self, args):
        self.args = args
        assert os.path.isdir(args.root_data_dir)
        self.parse_wiki_dump_tool = ParseWikiDumpTool(args)
        self.ent_name_id = self.parse_wiki_dump_tool.ent_name_id

        self.wiki = 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt'
        self.wiki_file = os.path.join(args.root_data_dir, self.wiki)
        assert os.path.isfile(self.wiki_file)

        self.out = 'generated/wikipedia_p_e_m.txt'
        self.out_file = os.path.join(args.root_data_dir, self.out)

        self.wiki_e_m_counts = self.load_from_wiki()
        self.write_p_e_m()

    @property
    def dict(self):
        return self.wiki_e_m_counts

    def load_from_wiki(self):
        num_lines = 0
        parsing_errors = 0
        list_ent_errors = 0
        diez_ent_errors = 0
        disambiguation_ent_errors = 0
        num_valid_hyperlinks = 0

        wiki_e_m_counts = dict()

        print('====> Computing Wikipedia p_e_m')

        with open(self.wiki_file, 'r') as reader:
            for line in reader:
                # line = line.rstrip('\n')
                num_lines = num_lines + 1
                if num_lines % 5000000 == 0:
                # **YD** debug the diff
                #if num_lines < 2829680:
                #    continue
                #print(repr(line))
                #if num_lines % 1 == 0:

                    print(
                        json.dumps(
                            {
                                'Processed lines': num_lines,
                                'Parsing errs': parsing_errors,
                                'List ent errs': list_ent_errors,
                                'diez errs': diez_ent_errors,
                                'disambig errs': disambiguation_ent_errors,
                                'Num valid hyperlinks': num_valid_hyperlinks,
                            },
                            indent=4,
                        )
                    )


                # **YD** debug the diff
                #if num_lines == 2829690:
                #    break

                if '<doc id="' not in line:
                    # **YD** extract_text_and_hyp not implemented
                    list_hyp, text, le_errs, p_errs, dis_errs, diez_errs = \
                        self.parse_wiki_dump_tool.extract_text_and_hyp(line, False)
                    parsing_errors += p_errs
                    list_ent_errors += le_errs
                    disambiguation_ent_errors += dis_errs
                    diez_ent_errors += diez_errs

                    for cnt, el in list_hyp.items():
                        # -- A valid(entity, mention) pair --
                        num_valid_hyperlinks = num_valid_hyperlinks + 1

                        mention = el['mention']
                        ent_wikiid = el['ent_wikiid']

                        if mention not in wiki_e_m_counts:
                            wiki_e_m_counts[mention] = dict()

                        if ent_wikiid in wiki_e_m_counts[mention]:
                            wiki_e_m_counts[mention][ent_wikiid] += 1
                        else:
                            wiki_e_m_counts[mention][ent_wikiid] = 1

        print('  -------> Done computing Wikipedia p(e|m). Num valid hyperlinks = ' + str(num_valid_hyperlinks))

        return wiki_e_m_counts

    # **YD** store in mention perline, may consider storing whole thing in a dictionary
    def write_p_e_m(self):
        print('====> Now sorting and writing p_m_e_from_wiki..')
        with open(self.out_file, 'w') as writer:
            for mention in self.wiki_e_m_counts:
                el_list = self.wiki_e_m_counts[mention]
                sorted_dict_items = sorted(el_list.items(), key=lambda x: x[1], reverse=True)
                s = ''
                total_freq = 0
                for wikiid, freq in sorted_dict_items:
                    s += '{}'.format(wikiid) + ',' + str(freq) + ','
                    # **YD** get_ent_name_from_wikiid not implemented
                    s += self.ent_name_id.get_ent_name_from_wikiid(wikiid).replace(' ', '_') + '\t'
                    total_freq += freq

                s = mention + '\t' + str(total_freq) + '\t' + s + '\n'
                writer.write(s)

        print('    Done sorting and writing.')


def test(args):
    gen_p_e_m_from_wiki = GenPEMFromWiki(args)
    return gen_p_e_m_from_wiki


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
    gen_p_e_m_from_wiki = test(args)
