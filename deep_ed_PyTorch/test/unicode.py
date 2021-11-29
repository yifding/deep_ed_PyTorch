a_str = 'ف'
a_unicode = '\u0641'
a_with_backsplash = '\\u0641'

assert a_str == a_unicode


def utf8_to_str_1(s='\\u0641'):
    assert len(s) == 6 and s[:2] == '\\u'
    # method 1:
    return chr(int(s[2:], base=16))


def utf8_to_str_2(s='\\u0641'):
    assert len(s) == 6 and s[:2] == '\\u'
    # method 2:
    return s.encode().decode("unicode-escape")


assert utf8_to_str_1(a_with_backsplash) == utf8_to_str_2(a_with_backsplash) == a_str == a_unicode

# print(utf8_to_str_1())
# print(utf8_to_str_2())
