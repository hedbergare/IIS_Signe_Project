
import torch
from torch import nn
from torchvision import  transforms
import dsntnn


# mellan varje nivå i u net har man ett
class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UNetModelDNNST(nn.Module):
    def __init__(
        self, encodingLayersSizes: list[list[int]], 
        decodingLayesSizes: list[list[int]], ):
        super().__init__()
    
        self.encodingLayersSizes = encodingLayersSizes
        self.decodingLayesSizes = decodingLayesSizes
        self.encBlocks = nn.ModuleList()
        self.decBlocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.maxPool=nn.MaxPool2d(2)


        #this is from the dnsf example i found i think 20 is refering to the amount of poition to identify
        self.hm_conv = nn.Conv2d(self.decodingLayesSizes[-1][1], 20, kernel_size=1, bias=False)



        for layer in self.encodingLayersSizes:
            self.encBlocks.append(Block(layer[0], layer[1]))

        for layer in self.decodingLayesSizes:
            self.decBlocks.append(Block(layer[0], layer[1]))
            self.upconvs.append(nn.ConvTranspose2d(layer[0], layer[1], 2, 2))

        





    def cropthing(self, encodingFeature,x):
        encodingFeature=transforms.CenterCrop([x.shape()[2] ,x.shape()[3]])(encodingFeature)
        return encodingFeature

    def forward(self, x):
        encodingFeatures=[]
        for layer in self.encBlocks:
            x = layer(x)
            encodingFeatures.append(x)
            x = self.maxPool(x)
        
        for layer, in zip(self.decBlocks,self.upconvs):
            x=layer[1](x)
            encodingFeature=encodingFeatures.pop()
            encodingFeature=self.cropthing(self, encodingFeature,x)
            x = torch.cat([x, encodingFeature], dim=1)
            x=layer[0](x)
        

        
        
        #final layer uses dnfs based on ruff interpetation  of code from gihhub see https://github.com/anibali/dsntnn/blob/master/examples/basic_usage.md
        x = self.hm_conv(x)
        x = dsntnn.flat_softmax(x)
        x = dsntnn.dsnt(x)
        
        
        return x


