import torch
from torch import nn
from testUnet import UNetModel
from ny_ladda_data import ladda
from torch import nn
import dsntnn
import matplotlib.pyplot as plt




x = UNetModel(4, 3, 20, 64)
optimizer = torch.optim.RMSprop(x.parameters(), lr=0.1)




loss = 0
num_epochs = 1
num_runs_per_backward = 2 
x.train()
num_divide_data = 50


for epoch in range(num_epochs):
    percent_fetched = 0
    for data_round in range(num_divide_data):
        print('Collect Data')
        train_data = ladda(typ='träning', seed=2, validation_round=1, data_divided_into=num_divide_data, data_pass=data_round)
        print('Data Collected')
        for frame in train_data:
            input=frame[0]
            val=frame[1]
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

            # Update model parameters with RMSprop
            optimizer.step()
            plt.imshow(heatmaps[0, 0].detach().numpy())
            plt.show()
        
           



torch.save(x.state_dict(), "landmark_model.pth")



