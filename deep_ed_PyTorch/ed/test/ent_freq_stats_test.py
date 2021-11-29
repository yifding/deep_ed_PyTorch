# -- Statistics of annotated entities based on their frequency in Wikipedia corpus
# -- Table 6 (left) from our paper


def ent_freq_to_key(f):
    # assert type(f) is int, "wrong type for f"
    if f == 0:
        return '0'
    elif f == 1:
        return '1'
    elif 2 <= f <= 5:
        return '2-5'
    elif 6 <= f <= 10:
        return '6-10'
    elif 11 <= f <= 20:
        return '11-20'
    elif 21 <= f <= 50:
        return '21-50'
    else:
        assert f > 50
        return '50+'


def new_ent_freq_map():
    m = {
            '0': 0.0,
            '1': 0.0,
            '2-5': 0.0,
            '6-10': 0.0,
            '11-20': 0.0,
            '21-50': 0.0,
            '50+': 0.0,
    }

    return m


def add_freq_to_ent_freq_map(m, f):
    m[ent_freq_to_key(f)] += 1


def print_ent_freq_maps_stats(smallm, bigm):
    print(' ===> entity frequency stats :')
    # **YD** not sure about the type of smallm and bigm
    for k in smallm:
        perc = 100.0 * smallm[k] / bigm[k] if bigm[k] > 0 else 0.0
        assert perc <= 100.0

        print(
                'freq = ' + '{}'.format(k) +
                ' : num = ' + '{}'.format(bigm[k]) +
                ' ; correctly classified = ' + '{}'.format(smallm[k]) +
                ' ; perc = ' + '{:.2f}'.format(perc)
        )
