reader = open('test_lua.log', 'r')
line = reader.readline()
while line:
    print(repr(line))
    line = reader.readline()

assert '\x0c' == '\f'
assert '\x0b' == '\v'
assert '\x0a' == '\n'
assert '\x0d' == '\r'

assert ' '.isspace()
assert '\f'.isspace()
assert '\v'.isspace()
assert '\n'.isspace()
assert '\r'.isspace()
assert '\t'.isspace()
