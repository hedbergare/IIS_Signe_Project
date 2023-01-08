import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image



# mellan varje nivå i u net har man ett
class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UNetModelFlatten(nn.Module):
    def __init__(
            self, depth, input_channels, output_channels, block_factor):
        super().__init__()

        self.blockFactor = block_factor
        self.depth = depth

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encBlocks = nn.ModuleList()
        self.decBlocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.maxPool = nn.ModuleList()
        self.encodingDropoutList = nn.ModuleList()
        self.decodingDropoutList = nn.ModuleList()

        self.linear11 = nn.Linear(640*480, 160)
        self.linear12 = nn.Linear(160, 2)
        self.linear21 = nn.Linear(640*480, 160)
        self.linear22 = nn.Linear(160, 1)
        self.relu = nn.ReLU()

        self.current_layer = 1
        for layer in range(self.depth):
            if self.current_layer == 1:
                self.encBlocks.append(
                    Block(self.input_channels, self.blockFactor))
            else:
                self.encBlocks.append(Block(
                    self.blockFactor*2**(self.current_layer-2), self.blockFactor*2**(self.current_layer-1)))
            if self.current_layer != self.depth-1:
                self.maxPool.append(nn.MaxPool2d(2))
            self.current_layer += 1

        self.current_layer -= 1
        for layer in range(self.depth-1):
            self.current_layer -= 1
            if self.current_layer == 1:
                self.decBlocks.append(
                    Block(self.blockFactor*2, self.output_channels))
            else:
                self.decBlocks.append(Block(
                    self.blockFactor*2**(self.current_layer), self.blockFactor*2**(self.current_layer-1)))

            self.upconvs.append(nn.ConvTranspose2d(
                self.blockFactor*2**(self.current_layer), self.blockFactor*2**(self.current_layer-1), 2, 2))

    def forward(self, x):
        encodingFeatures = []
        for i in range(len(self.encBlocks)):
            x = self.encBlocks[i](x)
            if i != (len(self.encBlocks)-1):
                encodingFeatures.append(x)
                x = self.maxPool[i](x)

        for i, dec in enumerate(self.decBlocks):
            x = self.upconvs[i](x)
            x = torch.cat((x, encodingFeatures[-1]))
            encodingFeatures.pop()
            x = dec(x)

        x = torch.flatten(x, start_dim=1)
        x_cord = self.linear11(x)
        x_cord = self.relu(x_cord)
        x_cord = self.linear12(x_cord)
        return x_cord