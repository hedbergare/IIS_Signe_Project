from testUnet import UNetModel
from ny_ladda_data import ladda
from torch import nn
import torch
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = UNetModel(4, 3, 20, 64).to(device)
x.load_state_dict(torch.load('landmark_model.pth'))
train_data = ladda(typ='träning', seed=2, validation_round=1, data_divided_into=100, data_pass=1)
x.eval()
input=train_data[0][0].to(device)
val=train_data[0][1]
cord,heatmap = x(input)
print(val)
print(cord)





