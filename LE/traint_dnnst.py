import torch
from torch import nn
from testUnet import UNetModel
from ny_ladda_data import ladda
from torch import nn
import dsntnn
import matplotlib.pyplot as plt
import sys
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
x = UNetModel(4, 3, 20, 64).to(device)

optimizer = torch.optim.RMSprop(x.parameters())


loss = 0
num_epochs = 3
num_runs_per_backward = 2 
x.train()
num_divide_data = 20

losses=[]
print(torch.__version__)
for epoch in range(num_epochs):
    print(epoch)
    percent_fetched = 0
    for data_round in range(num_divide_data):
        thetime=time.perf_counter()
    
        print('Collect Data')
        train_data = ladda(typ='träning', seed=2, validation_round=1, data_divided_into=num_divide_data, data_pass=data_round)
        print('Data Collected')
        for frame in train_data:
            input=frame[0].to(device)
            val=frame[1].to(device)
            val = torch.unsqueeze(val, dim=0)



            coords, heatmaps = x(input)

            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, val)
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, val, sigma_t=1.0)
            # Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)

            # Calculate gradients
            optimizer.zero_grad()
            loss.backward()
            del input
            del val
            # Update model parameters with RMSprop
        print(time.perf_counter()-thetime)
        print(data_round)
        torch.save(x.state_dict(), "landmark_model.pth")
           





