import json

d = dict()
d['a'] = {'1': float(1/3)}
d['b'] = {'a':'b'}

output = 'json_test.json'
with open(output, 'w') as json_file:
  json.dump(d, json_file)

with open(output, 'r') as json_file:
    new_d = json.load(json_file)

print(d)
print(new_d)
print(d == new_d)
#assert 0 == 1, "just a test"