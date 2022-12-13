import torch
from torch import nn
from Unet_flatten import UNetModel
from ny_ladda_data import ladda
from loss_function import distance_loss, binary_loss
from real_loss import real_loss
import math

device = torch.device('mps')

x = UNetModel(4, 3, 20, 64).to(device)




loss = 0
num_epochs = 1
num_runs_per_backward = 2 
optim = torch.optim.SGD(x.parameters(), lr=0.000000001, momentum=0.9)
x.train()
loss = 0
true_loss = 0

num_divide_data = 50


for epoch in range(num_epochs):
    percent_fetched = 0
    for data_round in range(num_divide_data):
        print('Collect Data')
        train_data = ladda(typ='träning', seed=2, validation_round=1, data_divided_into=num_divide_data, data_pass=data_round)
        print('Data Collected')
        for i in range(len(train_data)):
            output_cord, output_exists = x(train_data[i][0].to(device))
            output_cord, output_exists = x(train_data[i][0].to(device))
            exist_real = []
            for n in range(len(output_cord)):
                if train_data[i][1][n][0] == 0:
                    exist_real.append([0, 1])
                else:
                    exist_real.append([1, 0])
            exist_real = torch.tensor(exist_real)

            loss += distance_loss(output_cord, train_data[i][1].to(device))
            loss += binary_loss(output_exists, exist_real)

            true_loss += real_loss(output_cord, output_exists, train_data[i][1])


            if i % (num_runs_per_backward) == (num_runs_per_backward-1) or i == (len(train_data)-1):
                print(data_round,i, true_loss/num_runs_per_backward)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss = 0
                true_loss = 0

torch.save(x.state_dict(), "landmark_model.pth")
