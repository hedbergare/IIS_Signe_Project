import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import torchvision
import dsntnn


# mellan varje nivå i u net har man ett
class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UNetModel(nn.Module):
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
        self.hm_conv = nn.Conv2d(self.output_channels,
                                 20, kernel_size=1, bias=False)

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

        # kan inte komma på hur man ska göra sista ouputlayer
        self.finalayers = None

    def cropthing(self, encodingFeature, x):
        toPil = T.ToPILImage()
        image = toPil(encodingFeature)
        transform = T.CenterCrop((x.shape[1], x.shape[2]))
        image_crop = transform(image)
        toTensor = T.ToTensor()
        return toTensor(image_crop)

    def cropthing2(self, encodingFeature, x):
        size = (x.shape[1], x.shape[2])
        crop = torchvision.transforms.CenterCrop(size)
        for channel in range(len(encodingFeature)):
            crop(encodingFeature[channel])
        return encodingFeature

    def forward(self, x):
        encodingFeatures = []
        for i in range(len(self.encBlocks)):
            x = self.encBlocks[i](x)
            if i != (len(self.encBlocks)-1):
                encodingFeatures.append(x)
                x = self.maxPool[i](x)

        for i, dec in enumerate(self.decBlocks):
            print(i)
            x = self.upconvs[i](x)
            print('innan cat', x.shape)
            print(encodingFeatures[-1].shape)
            x = torch.cat((x, encodingFeatures[-1]))
            print('efter cat', x.shape)
            encodingFeatures.pop()
            print('Innan dec')
            x = dec(x)
            print('Eftert dec')

        x = self.hm_conv(x)
        x = dsntnn.flat_softmax(x)
        x = dsntnn.dsnt(x)

        return x
