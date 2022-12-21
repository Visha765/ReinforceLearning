import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from critic import CriticNet
from actor import ActorNet

inp = torch.rand(3, 2, requires_grad=True) 
target = torch.rand(3, 1)

model = ActorNet(2,1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

optimizer.zero_grad()     # すべてのパラメーターの勾配をゼロにします（初期化）

for i in range(1):
  out = model(inp)
  print(out)
  loss = torch.mean(-out)
  # loss = criterion(out, target)
  loss.backward()
  optimizer.step()
  # print(loss)
# print(loss)




# print(out.size(), out)

print(model)
l1 = model.stack[0].bias.grad
print(model.stack[0].bias.grad.requires_grad)


# target = torch.rand(3,9)
# criterion =  nn.CrossEntropyLoss()

# loss = criterion(out, target)
# print(loss)

# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# print(model)
# params = list(model.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

