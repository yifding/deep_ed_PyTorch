import os
import copy
import torch
import argparse

# **YD** keep in python, doesn't need to the same (can not now for my level)

from deep_ed_PyTorch.utils import modify_uppercase_phrase, upper, lower


class YagoCrosswikisWiki(object):
    def __init__(self, args):
        self.args = args

        self.crosswikis = 'generated/crosswikis_wikipedia_p_e_m.txt'
        self.crosswikis_file = os.path.join(args.root_data_dir, self.crosswikis)

        self.yago = 'generated/yago_p_e_m.txt'
        self.yago_file = os.path.join(args.root_data_dir, self.yago)

        self.pt = 'generated/ent_p_e_m_index.pt'
        self.pt_file = os.path.join(args.root_data_dir, self.pt)

        if os.path.exists(self.pt_file):
            self.load_from_pt()
        else:
            self.ent_p_e_m_index = dict()
            self.mention_lower_to_one_upper = dict()
            self.mention_total_freq = dict()

            self.load_from_crosswikis()
            self.load_from_yago()
            self.store_to_pt()

        print('    Done loading index')

    @property
    def dict(self):
        return self.ent_p_e_m_index

    def store_to_pt(self):
        print('====> Storing pt file: ' + self.pt_file)
        store_dict = dict()
        store_dict['ent_p_e_m_index'] = self.ent_p_e_m_index
        store_dict['mention_lower_to_one_upper'] = self.mention_lower_to_one_upper
        store_dict['mention_total_freq'] = self.mention_total_freq
        torch.save(store_dict, self.pt_file)

    def load_from_pt(self):
        print('====> load from pt file: ' + self.pt_file)
        store_dict = torch.load(self.pt_file)

        self.ent_p_e_m_index = store_dict['ent_p_e_m_index']
        self.mention_lower_to_one_upper = store_dict['mention_lower_to_one_upper']
        self.mention_total_freq = store_dict['mention_total_freq']

    def preprocess_mention(self, m):
        assert len(self.ent_p_e_m_index) > 0
        assert len(self.mention_lower_to_one_upper) > 0
        assert len(self.mention_total_freq) > 0

        cur_m = modify_uppercase_phrase(m)
        if cur_m not in self.ent_p_e_m_index:
            cur_m = m

        if self.mention_total_freq.get(m, 0) > self.mention_total_freq.get(cur_m, 0):
            cur_m = m

        if cur_m not in self.ent_p_e_m_index and lower(cur_m) in self.mention_lower_to_one_upper:
            cur_m = self.mention_lower_to_one_upper[lower(cur_m)]

        return cur_m

    def load_from_crosswikis(self):
        print('==> Loading crosswikis_wikipedia from file ', self.crosswikis_file)

        num_lines = 0
        with open(self.crosswikis_file, 'r') as reader:
            for line in reader:
                num_lines += 1

                if num_lines % 2000000 == 0:
                    print('Processed ' + str(num_lines) + ' lines. ')

                line = line.rstrip('\n\t')
                parts = line.split('\t')

                mention = parts[0]
                total = float(parts[1])

                """
                # **YD** DEBUG
                if mention != 'Washington':
                    continue
                else:
                    print(line)
                """

                if total >= 1:
                    if mention not in self.ent_p_e_m_index:
                        self.ent_p_e_m_index[mention] = dict()
                    else:
                        raise ValueError('repeated mention in crosswikis loading!')

                    self.mention_lower_to_one_upper[lower(mention)] = mention
                    self.mention_total_freq[mention] = total

                    for ent_part in parts[2:]:
                        ent_str = ent_part.split(',')
                        assert len(ent_str) >= 3, "length of entity string splitted by ',' less than 3!"

                        ent_wikiid = int(ent_str[0])
                        freq = float(ent_str[1])

                        assert ent_wikiid >= 1, "invalid ent_wikiid" + '{}'.format(ent_wikiid)
                        assert freq > 0, "frequency less or equal than one" + str(freq)

                        if ent_wikiid not in self.ent_p_e_m_index[mention]:
                            self.ent_p_e_m_index[mention][ent_wikiid] = float(freq / (total + 0.0))
                        else:
                            raise ValueError('repeated entities in the same crosswikis mention!')

    def load_from_yago(self):
        print('==> Loading yago from file ', self.yago_file)

        num_lines = 0
        with open(self.yago_file, 'r') as reader:
            for line in reader:
                num_lines += 1

                if num_lines % 2000000 == 0:
                    print('Processed ' + str(num_lines) + ' lines. ')

                line = line.rstrip('\n\t')
                parts = line.split('\t')

                mention = parts[0]
                total = float(parts[1])

                if total >= 1:
                    self.mention_lower_to_one_upper[lower(mention)] = mention

                    if mention not in self.mention_total_freq:
                        self.mention_total_freq[mention] = total
                    else:
                        self.mention_total_freq[mention] += total

                    yago_ment_ent_idx = dict()

                    # num_parts = len(parts)
                    for ent_part in parts[2:]:
                        ent_str = ent_part.split(',')
                        assert len(ent_str) >= 2, "length of entity string splitted by ',' less than 2!"

                        ent_wikiid = int(ent_str[0])
                        freq = 1

                        assert ent_wikiid >= 1, "invalid ent_wikiid" + '{}'.format(ent_wikiid)
                        # assert freq == 1, "frequency is not equal than one in yago" + str(freq)

                        if ent_wikiid not in yago_ment_ent_idx:
                            yago_ment_ent_idx[ent_wikiid] = float(freq / (total + 0.0))
                        else:
                            raise ValueError('repeated entities in the yago mention loading!')

                    if mention not in self.ent_p_e_m_index:
                        self.ent_p_e_m_index[mention] = copy.deepcopy(yago_ment_ent_idx)
                    else:
                        for ent_wikiid in yago_ment_ent_idx:
                            if ent_wikiid in self.ent_p_e_m_index[mention]:
                                self.ent_p_e_m_index[mention][ent_wikiid] += yago_ment_ent_idx[ent_wikiid]
                                if self.ent_p_e_m_index[mention][ent_wikiid] > 1.0:
                                    self.ent_p_e_m_index[mention][ent_wikiid] = 1.0
                            else:
                                self.ent_p_e_m_index[mention][ent_wikiid] = yago_ment_ent_idx[ent_wikiid]

        assert 'Dejan Koturovic' in self.ent_p_e_m_index
        assert 'Jose Luis Caminero' in self.ent_p_e_m_index


def test(args):
    YagoCrosswikisWiki(args)


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
