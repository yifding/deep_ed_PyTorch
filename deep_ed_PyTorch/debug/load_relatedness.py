import os
import json

file = '/scratch365/yding4/deep_ed_PyTorch/data/basic_data/relatedness/validate.svm'
"""
id = dict()
with open(file, 'r') as reader:
    for index, line in enumerate(reader):
        line = line.rstrip('\t\n')
        parts = line.split(' ')
        #print(parts[1])
        t = parts[1].split(':')
        p = t[1]
        print(p)
        if p not in id:
            id[p] = 1

print(len(id))
"""


def load_reltd_set(json_file, txt_file, set_type='validate'):
    print('==> Loading relatedness ' + set_type)
    if not os.path.isfile(json_file):
        print('  ---> json file NOT found. Loading relatedness ' + set_type + ' from txt file instead (slower).')
        reltd = dict()
        with open(txt_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\t\n')
                parts = line.split(' ')
                label = parts[0]
                assert int(label) == 0 or int(label) == 1
                t = parts[1].split(':')
                q = t[1]
                print(q)
                i = 1
                while parts[i] != '#':
                    i += 1
                i += 1

                ents = parts[i].split('-')
                e1 = ents[0]
                e2 = ents[1]

                if q not in reltd:
                    print('new_q: ', q)
                    reltd[q] = dict()
                    reltd[q]['e1'] = e1
                    reltd[q]['cand'] = dict()

                reltd[q]['cand'][e2] = label

        print('    Done loading relatedness ' + set_type +
              '. Num queries = ' + str(len(reltd)) + '\n')
        print('Writing json File for future usage. Next time relatedness dataset will load faster!')
        with open(json_file, 'w') as writer:
            json.dump(reltd, writer)
        print('    Done saving.')
        return reltd

    else:
        with open(json_file, 'r') as reader:
            reltd = json.load(reader)
        return reltd

file = '/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/basic_data/relatedness/validate.svm'
json_file = 'validate.json'
load_reltd_set(json_file, file, set_type='validate')
