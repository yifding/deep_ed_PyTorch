import torch

path = 'test.ckpt'
a = {'123': 123, 456: 123.524}
torch.save(a, path)
b = torch.load(path)
assert a == b