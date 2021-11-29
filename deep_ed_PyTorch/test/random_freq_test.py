import bisect
import random

# tiny example of the random logic:
word2id = {
    'UNK_W': 1,
    'a': 2,
    'b': 3,
    'c': 4,
}

id2word = {
    1: 'UNK_W',
    2: 'a',
    3: 'b',
    4: 'c',
}

w_f_start = {
    2: 0,
    3: 1,
    4: 3,
}

total_freq = 7

word_freq_select = list(w_f_start.values()) + [total_freq]
assert word_freq_select == [0, 1, 3, 7]


# k = random.randint(low=1, high=total_freq)

def rand_id(k, total_freq, word_freq_select):
    return bisect.bisect_left(word_freq_select, k) + 1


assert rand_id(1, total_freq, word_freq_select) == 2
assert rand_id(2, total_freq, word_freq_select) == 3
assert rand_id(3, total_freq, word_freq_select) == 3
assert rand_id(4, total_freq, word_freq_select) == 4
assert rand_id(5, total_freq, word_freq_select) == 4
assert rand_id(6, total_freq, word_freq_select) == 4
assert rand_id(7, total_freq, word_freq_select) == 4