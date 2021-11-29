import torch
import torch.nn as nn

target = torch.zeros(2, dtype=torch.long)
target[0] = 1
target[1] = 2

output = torch.zeros(2,3)

output[0][0] = 0.5
output[0][1] = 0.2
output[0][2] = 0.4

output[1][0] = 0.7
output[1][1] = 0.5
output[1][2] = 0.1

criterion = nn.MultiMarginLoss(p=1, weight=torch.ones(3), margin=0.1)
print(criterion(output, target))

# tensor(0.3167)


criterion = nn.MultiMarginLoss(p=1, weight=torch.ones(3), margin=0.1, reduction='none')
print(criterion(output, target))

# tensor([0.2333, 0.4000])