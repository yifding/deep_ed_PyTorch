# find the yago/crosswiki line which start with 'Washington'
py_crosswiki = '/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/crosswikis_wikipedia_p_e_m.txt'
py_yago = '/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/yago_p_e_m.txt'

st = 'Washington'
with open(py_crosswiki, 'r') as reader:
    for line in reader:
        parts = line.split('\t')[0]
        if parts == st:
            print(line)