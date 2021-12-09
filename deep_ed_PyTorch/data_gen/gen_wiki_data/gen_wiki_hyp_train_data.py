# -- Generate training data from Wikipedia hyperlinks by
# keeping the context and entity candidates for each hyperlink

# -- Format:
# -- ent_wikiid \t ent_name \t mention \t left_ctxt \t right_ctxt \t
# CANDIDATES \t[ent_wikiid, p_e_m, ent_name] + \t GT: \t pos, ent_wikiid, p_e_m, ent_name

import os
import argparse

from deep_ed_PyTorch import utils
from deep_ed_PyTorch.data_gen.parse_wiki_dump import ParseWikiDumpTool
from deep_ed_PyTorch.data_gen.indexes import YagoCrosswikisWiki


class GenWikiHypTrainData(object):
    def __init__(self, args):
        self.args = args

        self.ctxt_len = 100
        self.num_cand = 32

        self.wiki = 'basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt'
        self.wiki_file = os.path.join(args.root_data_dir, self.wiki)

        self.out = 'generated/wiki_hyperlink_contexts.csv'
        self.out_file = os.path.join(args.root_data_dir, self.out)

        self.parse_wiki_dump_tool = ParseWikiDumpTool(args)
        self.ent_name_id = self.parse_wiki_dump_tool.ent_name_id

        self.yago_crosswikis_wiki = YagoCrosswikisWiki(args)

        self.read_and_write()

    def read_and_write(self):
        ouf = open(self.out_file, "w")
        num_lines = 0
        num_valid_hyp = 0

        cur_words_num = 0
        cur_words = []
        cur_mentions = dict()
        cur_mentions_num = 0
        cur_ent_wikiid = -1

        with open(self.wiki_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\n\t')

                num_lines += 1
                if num_lines % 1000000 == 0:
                    print('Processed ' + str(num_lines) +  ' lines. Num valid hyp = ' + str(num_valid_hyp))

                if '<doc id="' not in line and '</doc>' not in line:
                    list_hyp, text, _, _, _, _ = self.parse_wiki_dump_tool.extract_text_and_hyp(line, True)
                    words_on_this_line = utils.split_in_words(text)
                    num_added_hyp = 0
                    line_mentions = dict()

                    for w in words_on_this_line:
                        wstart = w.startswith('MMSTART')
                        wend = w.startswith('MMEND')

                        if not wstart and not wend:
                            cur_words.append(w)
                            cur_words_num += 1

                        elif wstart:
                            mention_idx = w[len('MMSTART'):]
                            assert int(mention_idx) > 0, w
                            line_mentions[mention_idx] = {'start_off': cur_words_num + 1, 'end_off': -1}

                        elif wend:
                            num_added_hyp += 1
                            mention_idx = w[len('MMEND'):]
                            assert int(mention_idx) > 0, w
                            assert mention_idx in line_mentions
                            line_mentions[mention_idx]['end_off'] = cur_words_num

                    assert len(list_hyp) == num_added_hyp, line + ' :: ' + text + ' :: ' \
                            + str(num_added_hyp) + ' ' + str(len(list_hyp))

                    for hyp in list_hyp.values():
                        assert hyp['cnt'] in line_mentions

                        cur_mentions_num += 1
                        cur_mentions[str(cur_mentions_num)] = {
                            'mention': hyp['mention'],
                            'ent_wikiid': hyp['ent_wikiid'],
                            'start_off': line_mentions[hyp['cnt']]['start_off'],
                            'end_off': line_mentions[hyp['cnt']]['end_off'],
                        }

                elif '<doc id="' in line:

                    # -- Write results:
                    if cur_ent_wikiid != self.ent_name_id.unk_ent_wikiid and \
                            self.ent_name_id.is_valid_ent(cur_ent_wikiid):
                        header = str(cur_ent_wikiid) + '\t' + self.ent_name_id.get_ent_name_from_wikiid(cur_ent_wikiid) + \
                                 '\t'

                        for hyp in cur_mentions.values():
                            # **YD** ent_p_e_m_index not implemented
                            if hyp['mention'] in self.yago_crosswikis_wiki.ent_p_e_m_index \
                                    and len(self.yago_crosswikis_wiki.ent_p_e_m_index[hyp['mention']]) > 0:
                                assert len(hyp['mention']) > 0
                                s = header + hyp['mention'] + '\t'
                                left_ctxt = cur_words[max(0, hyp['start_off'] - self.ctxt_len): hyp['start_off']]

                                if len(left_ctxt) == 0:
                                    left_ctxt.append('EMPTYCTXT')

                                s += ' '.join(left_ctxt) + '\t'

                                right_ctxt = cur_words[hyp['end_off']: hyp['end_off'] + self.ctxt_len]
                                if len(right_ctxt) == 0:
                                    right_ctxt.append('EMPTYCTXT')

                                s += ' '.join(right_ctxt) + '\tCANDIDATES\t'

                                # -- Entity candidates from p(e | m) dictionary
                                sorted_cand = sorted(self.yago_crosswikis_wiki.ent_p_e_m_index[hyp['mention']].items(),
                                                     key=lambda x: x[1], reverse=True)[:self.num_cand]
                                candidate = []
                                gt_pos = -1

                                for index, (e, p) in enumerate(sorted_cand):
                                    candidate.append(str(e) + ',' + '{0:.3f}'.format(p) + ',' +
                                                     self.ent_name_id.get_ent_name_from_wikiid(e))

                                    # **YD** debug
                                    """
                                    import sys
                                    print('index: ', index)
                                    print('entity: ', type(e), e)
                                    print('p: ', type(p), p)
                                    sys.exit()
                                    """

                                    if e == hyp['ent_wikiid']:
                                        gt_pos = index

                                s += '\t'.join(candidate) + '\tGT:\t'

                                if gt_pos >= 0:
                                    num_valid_hyp += 1
                                    ouf.write(s + str(gt_pos) + ',' + candidate[gt_pos] + '\n')

                    cur_ent_wikiid = self.parse_wiki_dump_tool.extract_page_entity_title(line)
                    cur_words = []
                    cur_words_num = 0
                    cur_mentions = dict()
                    cur_mentions_num = 0

        ouf.close()
        print('    Done generating training data from Wiki dump. Num valid hyp = ' + str(num_valid_hyp))


def test(args):
    GenWikiHypTrainData(args)


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