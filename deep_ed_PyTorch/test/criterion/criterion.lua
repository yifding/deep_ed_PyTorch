require 'nn'


target = torch.zeros(1)
target[1] = 2

output = torch.zeros(1,3)

output[1][1] = 0.5
output[1][2] = 0.2
output[1][3] = 0.4


criterion = nn.MultiMarginCriterion(1, torch.ones(3), 0.1)

print(criterion:forward(output, target))



target = torch.zeros(1)
target[1] = 3

output = torch.zeros(1,3)

output[1][1] = 0.7
output[1][2] = 0.5
output[1][3] = 0.1


criterion = nn.MultiMarginCriterion(1, torch.ones(3), 0.1)

print(criterion:forward(output, target))



target = torch.zeros(2)
target[1] = 2
target[2] = 3

output = torch.zeros(2,3)

output[1][1] = 0.5
output[1][2] = 0.2
output[1][3] = 0.4

output[2][1] = 0.7
output[2][2] = 0.5
output[2][3] = 0.1

criterion = nn.MultiMarginCriterion(1, torch.ones(3), 0.1)

print(criterion:forward(output, target))