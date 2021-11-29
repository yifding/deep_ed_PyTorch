# **YD** ralated 'rewtr.reltd_ents_wikiid_to_rltdid' is currently not considered
# in dictionary, number is always stored with type(str), will be converted back after calculation

# basic_data/wiki_name_id_map.txt
# wikipedia-page  \t    wikiid

import os
import torch
import argparse

from deep_ed_PyTorch import utils
from deep_ed_PyTorch.data_gen.indexes import WikiDisambiguationIndex, WikiRedirectsPagesIndex

# **YD** consistent with relatedness rewtr
from deep_ed_PyTorch.entities.relatedness import REWTR

# **YD** preprocessing entity name process
"""
def preprocess_ent_name(ent_name, get_redirected_ent_title=True):
    ent_name = ent_name.strip()
    ent_name = ent_name.replace('&amp;', '&')
    ent_name = ent_name.repace('&quot;', '"')
    ent_name = ent_name.replace('_', ' ')
    # **YD** utils add funtion "first_letter_to_uppercase(ent_name)"
    ent_name = utils.first_letter_to_uppercase(ent_name)

    # **YD** add get_redirected_ent_title
    if get_redirected_ent_title:
        ent_name = get_redirected_ent_title(ent_name)

    return ent_name
"""


class EntNameID(object):
    def __init__(self, args):
        self.args = args
        self.rltd_only = False
        if hasattr(args, 'entities') and args.entities != 'ALL':
            # **YD** assert(rewtr.reltd_ents_wikiid_to_rltdid, 'Import relatedness.lua before ent_name_id.lua')
            self.rltd_only = True

            self.rewtr = REWTR(args)

        self.unk_ent_wikiid = 1
        self.entity_wiki_txtfilename = os.path.join(args.root_data_dir, 'basic_data/wiki_name_id_map.txt')
        self.entity_wiki_torchfilename = os.path.join(args.root_data_dir, 'generated/ent_name_id_map.pt')

        if self.rltd_only:
            self.entity_wiki_torchfilename = os.path.join(args.root_data_dir, 'generated/ent_name_id_map_RLTD.pt')

        self.wiki_dis = WikiDisambiguationIndex(self.args)
        self.wiki_red = WikiRedirectsPagesIndex(self.args)

        self.e_id_name = None
        print('==> Loading entity wikiid - name map')
        if os.path.exists(self.entity_wiki_torchfilename):
            self.e_id_name = self.load_from_torch(self.entity_wiki_torchfilename)
        else:
            self.e_id_name = self.load_from_text(self.entity_wiki_txtfilename)

        if not self.rltd_only:
            self.unk_ent_thid = self.e_id_name['ent_wikiid2thid'][self.unk_ent_wikiid]
        else:
            self.unk_ent_thid = self.rewtr.reltd_ents_wikiid_to_rltdid[self.unk_ent_wikiid]

        print('    Done loading entity name - wikiid. Size thid index = ', self.get_total_num_ents())
        print('    Done loading entity name - wikiid. Size num2ent index = ', len(self.dict['ent_name2wikiid']))

    @property
    def ent_wikiid2name(self):
        return self.e_id_name['ent_wikiid2name']

    @property
    def ent_name2wikiid(self):
        return self.e_id_name['ent_name2wikiid']

    @property
    def ent_wikiid2thid(self):
        return self.e_id_name['ent_wikiid2thid']

    @property
    def ent_thid2wikiid(self):
        return self.e_id_name['ent_thid2wikiid']

    @staticmethod
    def load_from_torch(torch_file):
        print('  ---> from torch file: ' + torch_file)
        return torch.load(torch_file)

    @property
    def dict(self):
        return self.e_id_name

    @property
    def key(self):
        if self.rltd_only:
            return ['ent_wikiid2name', 'ent_name2wikiid']
        else:
            return ['ent_wikiid2name', 'ent_name2wikiid', 'ent_wikiid2thid', 'ent_thid2wikiid']

    def load_from_text(self, txt_file):
        print('  ---> torch file NOT found. Loading from disk (slower). Out f = ' + txt_file)
        print('    Still loading entity wikiid - name map ...')

        # map for entity name to entity wiki id
        txt_output = dict()
        txt_output['ent_wikiid2name'] = dict()
        txt_output['ent_name2wikiid'] = dict()

        if not self.rltd_only:
            txt_output['ent_wikiid2thid'] = dict()
            txt_output['ent_thid2wikiid'] = dict()

        cnt = -1
        # **YD** big change: thid start from 0!

        with open(txt_file, 'r') as reader:
            for line in reader:
                parts = line.rstrip('\n').split('\t')
                assert len(parts) == 2, "entity_wiki_text is not length 2"
                ent_name, ent_wikiid = parts[0], int(parts[1])
                if not self.wiki_dis.is_disambiguation(ent_wikiid):
                    # **YD** realted considered
                    if not self.rltd_only or ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid:
                    # if not self.rltd_only:
                        txt_output['ent_wikiid2name'][ent_wikiid] = ent_name
                        txt_output['ent_name2wikiid'][ent_name] = ent_wikiid

                    if not self.rltd_only:
                        cnt += 1
                        txt_output['ent_wikiid2thid'][ent_wikiid] = cnt
                        txt_output['ent_thid2wikiid'][cnt] = ent_wikiid

        if not self.rltd_only:
            cnt = cnt + 1
            txt_output['ent_wikiid2thid'][self.unk_ent_wikiid] = cnt
            txt_output['ent_thid2wikiid'][cnt] = self.unk_ent_wikiid

        txt_output['ent_wikiid2name'][self.unk_ent_wikiid] = 'UNK_ENT'
        txt_output['ent_name2wikiid']['UNK_ENT'] = self.unk_ent_wikiid

        torch.save(txt_output, self.entity_wiki_torchfilename)

        return txt_output

    # ------------------------ Functions for wikiids and names-----------------
    def get_map_all_valid_ents(self):
        m = dict()
        for ent_wikiid in self.dict['ent_wikiid2name']:
            m[ent_wikiid] = '1'

        return m

    def is_valid_ent(self, ent_wikiid):
        assert type(ent_wikiid) is int
        return ent_wikiid in self.dict['ent_wikiid2name']

    def get_ent_name_from_wikiid(self, ent_wikiid):
        assert type(ent_wikiid) is int
        if ent_wikiid > 0 and ent_wikiid in self.dict['ent_wikiid2name']:
            return self.dict['ent_wikiid2name'][ent_wikiid]

        return 'YD_None'

    def preprocess_ent_name(self, ent_name, get_redirected_ent_title=True, special=True):
        if special:
            # **YD** core processing step
            ent_name = ent_name.strip(' \v\t\n\r\f')
        else:
            ent_name = ent_name.strip()
        ent_name = ent_name.replace('&amp;', '&')
        ent_name = ent_name.replace('&quot;', '"')
        ent_name = ent_name.replace('_', ' ')
        # **YD** utils add funtion "first_letter_to_uppercase(ent_name)"
        ent_name = utils.first_letter_to_uppercase(ent_name)

        # **YD** add get_redirected_ent_title
        if get_redirected_ent_title:
            ent_name = self.wiki_red.get_redirected_ent_title(ent_name)

        return ent_name

    def get_ent_wikiid_from_name(self, ent_name, not_verbose=False):
        verbose = not not_verbose
        ent_name = self.preprocess_ent_name(ent_name)

        if not ent_name or ent_name not in self.dict['ent_name2wikiid']:
            if verbose:
                print('Entity ' + ent_name + ' not found.')
            return self.unk_ent_wikiid
        return self.dict['ent_name2wikiid'][ent_name]

    # ------------------------ Functions for thids and wikiids ------------------------
    # -- ent wiki id -> thid
    def get_thid(self, ent_wikiid):
        assert type(ent_wikiid) is int
        # **YD** related has been considered
        if self.rltd_only:
            if ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid:
                ent_thid = self.rewtr.reltd_ents_wikiid_to_rltdid[ent_wikiid]
            else:
                ent_thid = self.unk_ent_thid
        else:
            if ent_wikiid in self.ent_wikiid2thid:
                ent_thid = self.ent_wikiid2thid[ent_wikiid]
            else:
                ent_thid = self.unk_ent_thid

        return ent_thid

    def contains_thid(self, ent_wikiid):
        # **YD** related has been considered
        # logic considers unk_ent_wikiid, may not be right
        assert type(ent_wikiid) is int

        if self.rltd_only:
            return ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid and ent_wikiid != self.unk_ent_wikiid
        else:
            return ent_wikiid in self.ent_wikiid2thid and ent_wikiid != self.unk_ent_wikiid

    def get_total_num_ents(self):
        # **YD** related has been considered
        if self.rltd_only:
            assert len(self.rewtr.reltd_ents_wikiid_to_rltdid) == self.rewtr.num_rltd_ents
            return len(self.rewtr.reltd_ents_wikiid_to_rltdid)
        else:
            return len(self.ent_thid2wikiid)

    def get_wikiid_from_thid(self, ent_thid):
        # **YD** related has been considered
        assert type(ent_thid) is int

        if self.rltd_only:
            if ent_thid in self.rewtr.reltd_ents_rltdid_to_wikiid:
                ent_wikiid = self.rewtr.reltd_ents_rltdid_to_wikiid[ent_thid]
            else:
                ent_wikiid = self.unk_ent_wikiid
        else:
            if ent_thid in self.ent_thid2wikiid:
                ent_wikiid = self.ent_thid2wikiid[ent_thid]
            else:
                ent_wikiid = self.unk_ent_wikiid
        return ent_wikiid

    # -- tensor of ent wiki ids --> tensor of thids
    # **YD** transform wiki ids --> thids to perform integer index of embeddings

    def get_ent_thids(self, ent_wikiids_tensor):
        ent_thid_tensor = ent_wikiids_tensor.clone()
        if ent_wikiids_tensor.dim() == 2:
            for i in range(ent_wikiids_tensor.size(0)):
                for j in range(ent_wikiids_tensor.size(1)):
                    ent_thid_tensor[i][j] = self.get_thid(int(ent_wikiids_tensor[i][j]))

        elif ent_wikiids_tensor.dim() == 1:
            for i in range(ent_wikiids_tensor.size(0)):
                ent_thid_tensor[i] = self.get_thid(int(ent_wikiids_tensor[i]))

        else:
            raise ValueError('Tensor with > 2 dimentions not supported')

        return ent_thid_tensor


def test(args):
    ent_name_id = EntNameID(args)
    assert ent_name_id.get_ent_wikiid_from_name('Anarchism') == '12', 'wikiid-name dictionary build error'

    """
    print('symbol + ' + 'Ȣ', ent_name_id.get_ent_wikiid_from_name('Ȣ'))
    print('unicode + ' + 'Ȣ', ent_name_id.get_ent_wikiid_from_name('\u0222'))
    print('symbol + ' + 'æ', ent_name_id.get_ent_wikiid_from_name('æ'))
    print('unicode + ' + 'æ', ent_name_id.get_ent_wikiid_from_name('\u00e6'))
    print('unicode + ' + 'æ' + '184309', ent_name_id.get_ent_name_from_wikiid('184309'))
    """
    print('symbol + ' + '\u00a0 New Jersey', ent_name_id.get_ent_wikiid_from_name('\u00a0 New Jersey'))
    print('unicode + ' + repr('\u00a0 New Jersey') + '21648', repr(ent_name_id.get_ent_name_from_wikiid('21648')))

    return ent_name_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate the coverage percent of entity dictionary on evaluate GT entity'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
        help='Root path of the data, $DATA_PATH.',
    )

    parser.add_argument(
        '--entities',
        type=str,
        default='RLTD',
        choices=['RLTD', '4EX', 'ALL'],
        help='Set of entities for which we train embeddings: 4EX (tiny, for debug) |'
             ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
    )

    args = parser.parse_args()
    ent_name_id = test(args)
