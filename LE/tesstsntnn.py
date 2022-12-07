import torch
from torch import nn
from testUnet import UNetModel
from ladda_data import prepare_all_data
import dsntnn

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from torch import nn
from testUnet import UNetModel
from ladda_data import prepare_all_data
import dsntnn

device = "cuda" if torch.cuda.is_available() else "cpu"
train_data,test_data=prepare_all_data(0.02)

model = UNetModel(3, 3, 3, 64)
optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)

print(model(train_data[0][0]))

n_epochs = 1

for epoch in range(n_epochs):

    for data in train_data[0]:
        coords,heatmaps =model(data[0])
        euc_losses = dsntnn.euclidean_losses(coords, torch.unsqueeze(data[1], dim=0))

        reg_losses = dsntnn.js_reg_losses(heatmaps, torch.unsqueeze(data[1], dim=0), sigma_t=1.0)
        loss = dsntnn.average_loss(euc_losses + reg_losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print(model(train_data[0][0]))