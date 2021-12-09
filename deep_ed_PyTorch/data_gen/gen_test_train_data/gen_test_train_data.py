import os
import re
import html
import argparse

from deep_ed_PyTorch import utils
from deep_ed_PyTorch.data_gen.indexes import YagoCrosswikisWiki
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID


class GenTestTrain(object):
    def __init__(self, args):
        self.args = args
        self.ctxt_len = 100
        self.num_cand = 100

        self.ent_name_id = EntNameID(args)
        self.yago_crosswikis_wiki = YagoCrosswikisWiki(args)

        output_dir = os.path.join(self.args.root_data_dir, 'generated/test_train_data')
        os.makedirs(output_dir, exist_ok=True)

        # **YD** debug
        for dataset in args.datasets:
            self.gen_test_ace(dataset)
        self.gen_aida_train()
        self.gen_aida_test()

    def gen_test_ace(self, dataset='ace2004'):
        print('\nGenerating test data from ' + dataset + ' set ')

        path = os.path.join('basic_data/test_datasets/wned-datasets/', dataset)
        path = os.path.join(self.args.root_data_dir, path)

        out_file = os.path.join(self.args.root_data_dir, 'generated/test_train_data/wned-' + dataset + '.csv')
        anno_file = os.path.join(path, dataset + '.xml')

        print('input main path: ', path)
        print('anno_file: ', anno_file)
        print('out_file: ', out_file)

        writer = open(out_file, 'w')

        reader = open(anno_file, 'r')
        line = reader.readline()

        num_nonexistent_ent_id = 0
        num_correct_ents = 0

        cur_doc_text = ''
        cur_doc_name = ''

        while line:
            doc_str_start = 'document docName=\"'
            doc_str_end = '\">'

            if doc_str_start in line:
                start = line.find(doc_str_start)
                end = line.find(doc_str_end)
                cur_doc_name = line[start + len(doc_str_start): end]
                while '&amp;' in cur_doc_name:
                    cur_doc_name = cur_doc_name.replace('&amp;', '&')
                # cur_doc_name = html.unescape(cur_doc_name)
                cur_doc_text = ''
                with open(os.path.join(path, 'RawText/' + cur_doc_name), 'r') as raw_reader:
                    for txt_line in raw_reader:
                        cur_doc_text += txt_line.rstrip('\n') + ' '
                while '&amp;' in cur_doc_text:
                    cur_doc_text = cur_doc_text.replace('&amp;', '&')
                # cur_doc_text = html.unescape(cur_doc_text)

            else:
                if '<annotation>' in line:

                    line = reader.readline()
                    assert '<mention>' in line and '</mention>' in line
                    m_start = line.find('<mention>') + len('<mention>')
                    m_end = line.find('</mention>')
                    cur_mention = line[m_start: m_end]
                    while '&amp;' in cur_mention:
                        cur_mention = cur_mention.replace('&amp;', '&')

                    line = reader.readline()
                    # assert '<wikiName>' in line and '</wikiName>' in line
                    e_start = line.find('<wikiName>') + len('<wikiName>')
                    e_end = line.find('</wikiName>')
                    cur_ent_title = '' if '<wikiName/>' in line else line[e_start: e_end]

                    line = reader.readline()
                    assert '<offset>' in line and '</offset>' in line
                    off_start = line.find('<offset>') + len('<offset>')
                    off_end = line.find('</offset>')
                    offset = int(line[off_start: off_end])

                    line = reader.readline()
                    assert '<length>' in line and '</length>' in line
                    len_start = line.find('<length>') + len('<length>')
                    len_end = line.find('</length>')
                    # length = int(line[len_start: len_end])
                    length = len(cur_mention)

                    line = reader.readline()
                    if '<entity/>' in line:
                        line = reader.readline()

                    assert '</annotation>' in line

                    offset = max(0, offset - 10)
                    while cur_doc_text[offset: offset + length] != cur_mention:
                        offset = offset + 1

                        # **YD** debug
                        if offset >= len(cur_doc_text):
                            print(f"cur_mention:{cur_mention}; cur_ent_title:{cur_ent_title}")
                            print(repr(cur_doc_text))
                            import sys
                            sys.exit()

                    # **YD** rewrite the search logic, find the closest mention which has fewer or equal positions
                    # compared to labelled offset.
                    # candidate_offsets = [
                    #     m.start() for m in re.finditer(cur_mention, cur_doc_text)
                    # ]
                    # candidate_offsets = sorted(candidate_offsets, key=lambda x: abs(x-offset))
                    # assert len(candidate_offsets) > 0
                    # offset = candidate_offsets[0]

                    # **YD** preprocess_mention has been implemented
                    cur_mention = self.yago_crosswikis_wiki.preprocess_mention(cur_mention)

                    if cur_ent_title != 'NIL' and cur_ent_title != '' and len(cur_ent_title) > 0:
                        # **YD** get_ent_wikiid_from_name has been implemented
                        cur_ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(cur_ent_title)

                        # **YD** unk_ent_wikiid has been implemented
                        if cur_ent_wikiid == self.ent_name_id.unk_ent_wikiid:
                            num_nonexistent_ent_id += 1
                            print('unknown entity anno: ', cur_ent_title)
                        else:
                            num_correct_ents += 1

                        assert len(cur_mention) > 0

                        s = cur_doc_name + '\t' + cur_doc_name + '\t' + cur_mention + '\t'

                        # **YD** split_in_words has been implemented, should return a list
                        left_words = utils.split_in_words(cur_doc_text[0: offset])
                        left_ctxt = left_words[-self.ctxt_len:]
                        if len(left_ctxt) == 0:
                            left_ctxt.append('EMPTYCTXT')
                        s += ' '.join(left_ctxt) + '\t'

                        # **YD** split_in_words not implemented, should return a list
                        right_words = utils.split_in_words(cur_doc_text[offset + length:])
                        right_ctxt = right_words[:self.ctxt_len]
                        if len(right_ctxt) == 0:
                            right_ctxt.append('EMPTYCTXT')
                        s += ' '.join(right_ctxt) + '\tCANDIDATES\t'


                        # Entity candidates from p(e|m) dictionary
                        # **YD** ent_p_e_m_index has been implemented
                        if cur_mention in self.yago_crosswikis_wiki.ent_p_e_m_index and \
                                len(self.yago_crosswikis_wiki.ent_p_e_m_index[cur_mention]) > 0:
                            sorted_cand = sorted(self.yago_crosswikis_wiki.ent_p_e_m_index[cur_mention].items(),
                                                 key=lambda x: x[1], reverse=True)

                            candidates = []
                            gt_pos = -1

                            for index, (wikiid, p) in enumerate(sorted_cand[: self.num_cand]):
                                # **YD** get_ent_wikiid_from_name has been implemented
                                candidates.append(str(wikiid) + ',' + '{0:.3f}'.format(p) + ',' + \
                                                  self.ent_name_id.get_ent_name_from_wikiid(wikiid))

                                if wikiid == cur_ent_wikiid:
                                    # **YD** index is based on python array, start with 0
                                    gt_pos = index

                            s += '\t'.join(candidates) + '\tGT:\t'

                            if gt_pos >=0:
                                s += str(gt_pos) + ',' + candidates[gt_pos] + '\n'
                            else:
                                # **YD** unk_ent_wikiid has been implemented
                                if cur_ent_wikiid != self.ent_name_id.unk_ent_wikiid:
                                    s += '-1,' + str(cur_ent_wikiid) + ',' + cur_ent_title + '\n'

                                else:
                                    s += '-1\n'

                        else:
                            # **YD** unk_ent_wikiid has been not implemented
                            if cur_ent_wikiid != self.ent_name_id.unk_ent_wikiid:
                                s += 'EMPTYCAND\tGT:\t-1,' + str(cur_ent_wikiid) + ',' + cur_ent_title + '\n'
                            else:
                                s += 'EMPTYCAND\tGT:\t-1\n'

                        writer.write(s)

            line = reader.readline()

        writer.close()

        print('Done '+ dataset + '.')
        print('num_nonexistent_ent_id = ' + str(num_nonexistent_ent_id) + '; num_correct_ents = ' + str(num_correct_ents))

    def gen_aida_train(self):
        aida_train = 'basic_data/test_datasets/AIDA/aida_train.txt'
        aida_train_file = os.path.join(self.args.root_data_dir, aida_train)
        out = 'generated/test_train_data/aida_train.csv'
        out_file = os.path.join(self.args.root_data_dir, out)
        writer = open(out_file, 'w')

        num_nme = 0
        num_nonexistent_ent_title = 0
        num_nonexistent_ent_id = 0
        num_nonexistent_both = 0
        num_correct_ents = 0
        num_total_ents = 0

        cur_words_num = 0
        cur_words = []
        cur_mentions = dict()
        cur_mentions_num = 0
        cur_doc_name = ''

        def write_results():
            if cur_doc_name != '':
                header = cur_doc_name + '\t' + cur_doc_name + '\t'

                for hyp in cur_mentions.values():
                    assert len(hyp['mention']) > 0, line

                    s = header + hyp['mention'] + '\t'

                    # **YD** the ctxt logic should be double checked
                    left_ctxt = cur_words[max(hyp['start_off'] - self.ctxt_len, 0): hyp['start_off']]
                    if len(left_ctxt) == 0:
                        left_ctxt.append('EMPTYCTXT')

                    s += ' '.join(left_ctxt) + '\t'

                    right_ctxt = cur_words[hyp['end_off']: hyp['end_off'] + self.ctxt_len]
                    if len(right_ctxt) == 0:
                        right_ctxt.append('EMPTYCTXT')

                    s += ' '.join(right_ctxt) + '\tCANDIDATES\t'

                    # Entity candidates from p(e|m) dictionary
                    # **YD** ent_p_e_m_index not implemented
                    if hyp['mention'] in self.yago_crosswikis_wiki.ent_p_e_m_index and \
                            len(self.yago_crosswikis_wiki.ent_p_e_m_index[hyp['mention']]) > 0:
                        sorted_cand = sorted(self.yago_crosswikis_wiki.ent_p_e_m_index[hyp['mention']].items(),
                                             key=lambda x: x[1], reverse=True)

                        candidates = []
                        gt_pos = -1
                        for index, (wikiid, p) in enumerate(sorted_cand[: self.num_cand]):
                            if wikiid == hyp['ent_wikiid']:
                                gt_pos = index
                            candidates.append(str(wikiid) + ',' + '{0:.3f}'.format(p) + ',' + \
                                              self.ent_name_id.get_ent_name_from_wikiid(wikiid))

                        s += '\t'.join(candidates) + '\tGT:\t'
                        if gt_pos >= 0:
                            writer.write(s + str(gt_pos) + ',' + candidates[gt_pos] + '\n')

        with open(aida_train_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\n\t')
                if '-DOCSTART-' in line:
                    write_results()

                    # **YD** not sure the function here
                    cur_doc_name = line[12:]

                    cur_words = []
                    cur_words_num = 0
                    cur_mentions = dict()
                    cur_mentions_num = 0

                else:
                    parts = line.split('\t')
                    num_parts = len(parts)
                    # print(num_parts)
                    assert num_parts in [0, 1, 4, 6, 7], repr(line)
                    if num_parts > 0:
                        if num_parts == 4 and parts[1] == 'B':
                            num_nme += 1

                        if num_parts in [6, 7] and parts[1] == 'B':

                            # **YD** preprocess_mention not implemented
                            cur_mention = self.yago_crosswikis_wiki.preprocess_mention(parts[2])

                            x = parts[4].find('/wiki/')
                            y = x + len('/wiki/')
                            cur_ent_title = parts[4][y:]

                            # **YD** wiki id is always int in our implementation
                            cur_ent_wikiid = int(parts[5])

                            # **YD** get_ent_name_from_wikiid, get_ent_wikiid_from_name not implemented
                            index_ent_title = self.ent_name_id.get_ent_name_from_wikiid(cur_ent_wikiid)
                            index_ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(cur_ent_title)

                            final_ent_wikiid = index_ent_wikiid

                            # **YD** unk_ent_wikiid has been implemented
                            if final_ent_wikiid == self.ent_name_id.unk_ent_wikiid:
                                final_ent_wikiid = cur_ent_wikiid

                            if index_ent_title == cur_ent_title and cur_ent_wikiid == index_ent_wikiid:
                                num_correct_ents += 1
                            elif index_ent_title != cur_ent_title and cur_ent_wikiid != index_ent_wikiid:
                                num_nonexistent_both += 1
                            elif index_ent_title != cur_ent_title and cur_ent_wikiid == index_ent_wikiid:
                                num_nonexistent_ent_title += 1
                            elif index_ent_title == cur_ent_title and cur_ent_wikiid != index_ent_wikiid:
                                num_nonexistent_ent_id = num_nonexistent_ent_id + 1
                            else:
                                raise ValueError('logic ERROR!')

                            num_total_ents += 1

                            cur_mentions_num += 1
                            cur_mentions[cur_mentions_num] = dict()
                            cur_mentions[cur_mentions_num]['mention'] = cur_mention
                            cur_mentions[cur_mentions_num]['ent_wikiid'] = final_ent_wikiid
                            # **YD** careful to offset, may need to revise
                            cur_mentions[cur_mentions_num]['start_off'] = cur_words_num
                            cur_mentions[cur_mentions_num]['end_off'] = cur_words_num + len(parts[2].split(' '))

                        #  **YD** split_in_words not implemented
                        words_on_this_line = utils.split_in_words(parts[0])
                        for w in words_on_this_line:
                            # **YD** modify_uppercase_phrase not implemented
                            cur_words.append(utils.modify_uppercase_phrase(w))
                            cur_words_num += 1
            write_results()

        writer.close()

        print('    Done AIDA.')
        print('num_nme = ' + str(num_nme) + '; num_nonexistent_ent_title = ' + str(num_nonexistent_ent_title))
        print('num_nonexistent_ent_id = ' + str(num_nonexistent_ent_id) + '; num_nonexistent_both = ' +
              str(num_nonexistent_both))
        print('num_correct_ents = ' + str(num_correct_ents) + '; num_total_ents = ' + str(num_total_ents))

    def gen_aida_test(self):
        aida_test = 'basic_data/test_datasets/AIDA/testa_testb_aggregate_original'
        aida_test_file = os.path.join(self.args.root_data_dir, aida_test)
        assert os.path.isfile(aida_test_file)

        out_testA = 'generated/test_train_data/aida_testA.csv'
        out_testB = 'generated/test_train_data/aida_testB.csv'
        out_testA_file = os.path.join(self.args.root_data_dir, out_testA)
        out_testB_file = os.path.join(self.args.root_data_dir, out_testB)

        ouf_A = open(out_testA_file, 'w')
        ouf_B = open(out_testB_file, 'w')
        ouf = ouf_A


        num_nme = 0
        num_nonexistent_ent_title = 0
        num_nonexistent_ent_id = 0
        num_nonexistent_both = 0
        num_correct_ents = 0
        num_total_ents = 0

        cur_words_num = 0
        cur_words = []
        cur_mentions = dict()
        cur_mentions_num = 0
        cur_doc_name = ''

        def write_results():
            if cur_doc_name != '':
                header = cur_doc_name + '\t' + cur_doc_name + '\t'

                for hyp in cur_mentions.values():
                    assert len(hyp['mention']) > 0, line

                    s = header + hyp['mention'] + '\t'

                    # **YD** the ctxt logic should be double checked
                    left_ctxt = cur_words[max(hyp['start_off'] - self.ctxt_len, 0): hyp['start_off']]
                    if len(left_ctxt) == 0:
                        left_ctxt.append('EMPTYCTXT')

                    s += ' '.join(left_ctxt) + '\t'

                    right_ctxt = cur_words[hyp['end_off']: hyp['end_off'] + self.ctxt_len]
                    if len(right_ctxt) == 0:
                        right_ctxt.append('EMPTYCTXT')

                    s += ' '.join(right_ctxt) + '\tCANDIDATES\t'

                    # Entity candidates from p(e|m) dictionary
                    # **YD** ent_p_e_m_index not implemented
                    if hyp['mention'] in self.yago_crosswikis_wiki.ent_p_e_m_index and \
                            len(self.yago_crosswikis_wiki.ent_p_e_m_index[hyp['mention']]) > 0:
                        sorted_cand = sorted(self.yago_crosswikis_wiki.ent_p_e_m_index[hyp['mention']].items(),
                                             key=lambda x: x[1], reverse=True)

                        candidates = []
                        gt_pos = -1
                        for index, (wikiid, p) in enumerate(sorted_cand[: self.num_cand]):
                            if wikiid == hyp['ent_wikiid']:
                                gt_pos = index
                            candidates.append(str(wikiid) + ',' + '{0:.3f}'.format(p) + ',' + \
                                              self.ent_name_id.get_ent_name_from_wikiid(wikiid))

                        s += '\t'.join(candidates) + '\tGT:\t'
                        if gt_pos >= 0:
                            ouf.write(s + str(gt_pos) + ',' + candidates[gt_pos] + '\n')
                        else:
                            if hyp['ent_wikiid'] != self.ent_name_id.unk_ent_wikiid:
                                ouf.write(s + '-1,' + str(hyp['ent_wikiid']) + ','
                                          + self.ent_name_id.get_ent_name_from_wikiid(hyp['ent_wikiid']) + '\n')
                            else:
                                ouf.write(s + '-1\n')

                    else:
                        if hyp['ent_wikiid'] != self.ent_name_id.unk_ent_wikiid:
                            ouf.write(s + 'EMPTYCAND\tGT:\t-1,' + str(hyp['ent_wikiid']) + ','
                                      + self.ent_name_id.get_ent_name_from_wikiid(hyp['ent_wikiid']) + '\n')
                        else:
                            ouf.write(s + 'EMPTYCAND\tGT:\t-1\n')

        with open(aida_test_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\n\t')
                if '-DOCSTART-' in line:
                    write_results()

                    if 'testa' in cur_doc_name and 'testb' in line:
                        ouf = ouf_B
                        print('Done validation testA : ')
                        print('num_nme = ' + str(num_nme) + '; num_nonexistent_ent_title = ' +
                              str(num_nonexistent_ent_title))
                        print('num_nonexistent_ent_id = ' + str(num_nonexistent_ent_id) +
                            '; num_nonexistent_both = ' + str(num_nonexistent_both))
                        print('num_correct_ents = ' + str(num_correct_ents) +
                            '; num_total_ents = ' + str(num_total_ents))

                    words = utils.split_in_words(line)
                    for w in words:
                        if 'testa' in w or 'testb' in w:
                            cur_doc_name = w
                            break

                    cur_words_num = 0
                    cur_words = []
                    cur_mentions = dict()
                    cur_mentions_num = 0

                else:
                    parts = line.split('\t')
                    num_parts = len(parts)
                    assert num_parts in [0, 1, 4, 6, 7], line

                    if num_parts > 0:
                        if num_parts == 4 and parts[1] == 'B':
                            num_nme += 1

                        if num_parts in [6, 7] and parts[1] == 'B':

                            # **YD** preprocess_mention not implemented
                            cur_mention = self.yago_crosswikis_wiki.preprocess_mention(parts[2])

                            x = parts[4].find('/wiki/')
                            y = x + len('/wiki/')
                            cur_ent_title = parts[4][y:]

                            # **YD** wiki id is always int in our implementation
                            cur_ent_wikiid = int(parts[5])

                            # **YD** get_ent_name_from_wikiid, get_ent_wikiid_from_name not implemented
                            index_ent_title = self.ent_name_id.get_ent_name_from_wikiid(cur_ent_wikiid)
                            index_ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(cur_ent_title)

                            final_ent_wikiid = index_ent_wikiid

                            # **YD** unk_ent_wikiid not implemented
                            if final_ent_wikiid == self.ent_name_id.unk_ent_wikiid:
                                final_ent_wikiid = cur_ent_wikiid

                            if index_ent_title == cur_ent_title and cur_ent_wikiid == index_ent_wikiid:
                                num_correct_ents += 1
                            elif index_ent_title != cur_ent_title and cur_ent_wikiid != index_ent_wikiid:
                                num_nonexistent_both += 1
                            elif index_ent_title != cur_ent_title and cur_ent_wikiid == index_ent_wikiid:
                                num_nonexistent_ent_title += 1
                            elif index_ent_title == cur_ent_title and cur_ent_wikiid != index_ent_wikiid:
                                num_nonexistent_ent_id = num_nonexistent_ent_id + 1
                            else:
                                raise ValueError('logic ERROR!')

                            num_total_ents += 1

                            cur_mentions_num += 1
                            cur_mentions[cur_mentions_num] = dict()
                            cur_mentions[cur_mentions_num]['mention'] = cur_mention
                            cur_mentions[cur_mentions_num]['ent_wikiid'] = final_ent_wikiid
                            # **YD** careful to offset, may need to revise
                            cur_mentions[cur_mentions_num]['start_off'] = cur_words_num
                            cur_mentions[cur_mentions_num]['end_off'] = cur_words_num + len(parts[2].split(' '))

                        words_on_this_line = utils.split_in_words(parts[0])
                        for w in words_on_this_line:
                            cur_words.append(utils.modify_uppercase_phrase(w))
                            cur_words_num += 1

            write_results()

        ouf_A.close()
        ouf_B.close()

        print('    Done AIDA.')
        print('num_nme = ' + str(num_nme) + '; num_nonexistent_ent_title = ' + str(num_nonexistent_ent_title))
        print('num_nonexistent_ent_id = ' + str(num_nonexistent_ent_id) + '; num_nonexistent_both = ' +
              str(num_nonexistent_both))
        print('num_correct_ents = ' + str(num_correct_ents) + '; num_total_ents = ' + str(num_total_ents))


def test(args):
    GenTestTrain(args)


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


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


    parser.add_argument(
        '--datasets',
        type=eval,
        default="['ace2004', 'aquaint', 'clueweb', 'msnbc', 'wikipedia',]",
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    print(args)
    test(args)