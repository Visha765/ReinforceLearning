import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from critic import CriticNet

x = torch.rand(1, 2+1) 
model = CriticNet(2,1)
out = model(x)
print(out.size(), out)
print(model)
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

