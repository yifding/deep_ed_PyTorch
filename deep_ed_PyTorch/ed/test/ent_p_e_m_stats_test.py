def ent_prior_to_key(f):
    if 0 <= f <= 0.001:
        return '<=0.001'
    elif f <= 0.003:
        return '0.001-0.003'
    elif f <= 0.01:
        return '0.003-0.01'
    elif f <= 0.03:
        return '0.01-0.03'
    elif f <= 0.1:
        return '0.03-0.1'
    elif f <= 0.3:
        return '0.1-0.3'
    else:
        assert f > 0.3, 'invalid number of f'
        return '0.3+'


def new_ent_prior_map():
    m = {
        '<=0.001': 0.0,
        '0.001-0.003': 0.0,
        '0.003-0.01': 0.0,
        '0.01-0.03': 0.0,
        '0.03-0.1': 0.0,
        '0.1-0.3': 0.0,
        '0.3+': 0.0,
    }

    return m


def add_prior_to_ent_prior_map(m, f):
  m[ent_prior_to_key(f)] += 1


def print_ent_prior_maps_stats(smallm, bigm):
    print(' ===> entity p(e|m) stats :')
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
