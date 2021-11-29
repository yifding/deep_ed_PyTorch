import re
import torch


def lower(s):
    f = lambda pat: pat.group(1).lower()
    return re.sub(r'([A-Z])', f, s)


def upper(s):
    f = lambda pat: pat.group(1).upper()
    return re.sub(r'([a-z])', f, s)


def first_letter_to_uppercase(s, eng=True):
    if not s or len(s) == 0:
        return s

    if eng:
        return upper(s[0]) + s[1:]
    else:
        return s[0].upper() + s[1:]


def split_in_words(input_str):
    return re.findall(r'\w+', input_str)


def modify_uppercase_phrase(s):
        if s == upper(s):
            words = split_in_words(lower(s))
            res = []
            for word in words:
                res.append(first_letter_to_uppercase(word))
            return ' '.join(res)

        else:
            return s


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)