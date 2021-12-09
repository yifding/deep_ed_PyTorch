import os
import copy
# mention:
#   entity 1:
"""
class EntityAnno(object):
    def __init__(self, entity_name, wikiid, freq):
        self.entity_name = entity_name
        self.wikiid = wikiid
        self.freq = freq

class MentionAnno(object):
    def __init__(self, mention_name, total_freq, entity_anno=None):
        self.mention_name = mention_name
        self.total_freq = total_freq
        self.entity_anno = entity_anno if entity_anno else dict()

    def add_entity(self, entity_anno):
"""


def process_cross_dict(file):
    re_dict = dict()
    with open(file, 'r') as reader:
        for i, line in enumerate(reader):
            if i == 0:
                print(repr(line))
            original_line = copy.deepcopy(line)
            line = line.rstrip()
            parts = line.split('\t')
            mention = parts[0]
            total_freq = parts[1]

            if int(total_freq) < 0:
                print(repr(original_line))
                raise ValueError('total frequency less than 0!')

            if mention not in re_dict:
                re_dict[mention] = {"TOTAL_freq": total_freq}

            for ent_part in parts[2:]:
                ent_split_parts = ent_part.split(',')
                assert len(ent_split_parts) >= 3
                wikiid = ent_split_parts[0]
                freq = ent_split_parts[1]
                ent_name = ','.join(ent_split_parts[2:])
                if wikiid not in re_dict[mention]:
                    re_dict[mention][wikiid] = {'ent_name': ent_name, 'freq': freq}
                else:
                    assert 0, 'repeated entity in the mention annotation!'

    return re_dict


lua_cross = '/scratch365/yding4/EL_resource/data/deep_ed_data/generated/crosswikis_wikipedia_p_e_m.txt'
py_cross = '/scratch365/yding4/deep_ed_PyTorch/data/generated/crosswikis_wikipedia_p_e_m.txt'

assert os.path.isfile(lua_cross)
assert os.path.isfile(py_cross)

lua_dict = process_cross_dict(lua_cross)
print('load lua_dict DONE!')
py_dict = process_cross_dict(py_cross)
print('load py_dict DONE!')


print('length equal: ', len(lua_dict) == len(py_dict))

lua_dict_keys = sorted(lua_dict.keys())
py_dict_keys = sorted(py_dict.keys())
"""
if lua_dict_keys == py_dict_keys:
    print('mention keys equal: ', 'True')
else:
    print('mention keys equal: ', 'False')
    for lua_key, py_key in zip(lua_dict_keys, py_dict_keys):
        if lua_key != py_key:
            print('different mention dict key: lua, py', lua_key, py_key)
            break
"""
"""
for lua_key, py_key in zip(lua_dict_keys, py_dict_keys):
    if lua_dict[lua_key]['TOTAL_freq'] != py_dict[py_key]['TOTAL_freq']:
        print('freq different')
"""






