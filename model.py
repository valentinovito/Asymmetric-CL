import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

"""
Slightly adapted from the code for the SuperCon model: https://github.com/keyu07/SuperCon_ISIC.
During the first training stage, the architecture comprises of NetE_resnet and rep_heads.
During the second training stage, the architecture comprises of NetE_resnet (frozen) and NetC_resnet.
"""
class NetE_resnet(pl.LightningModule):

    def __init__(self, model):
        super(NetE_resnet, self).__init__()
        if model == 'resnet18':
            net = models.resnet18(pretrained=True)
            
        if model == 'resnet50':
            net = models.resnet50(pretrained=True)
            
        if model == 'resnet101':
            net = models.resnet101(pretrained=True)

        if model == 'resnet152':
            net = models.resnet152(pretrained=True)

        # Get the feature extractor
        self.netE = nn.Sequential(*list(net.children())[:-1])

    def forward(self, input):
        x = self.netE(input)
        
        out = torch.flatten(x, 1)
        return out

class NetC_resnet(pl.LightningModule):

    def __init__(self, model, output_dim=2):
        super(NetC_resnet, self).__init__()
        
        if model == 'resnet18':
            self.net = nn.Sequential(
                nn.Linear(512, output_dim)
            )

        else:
            self.net = nn.Sequential(
                nn.Linear(2048, output_dim)
            )
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        out = self.net(input)
        #out = self.sigmoid(out)
        return out

class rep_heads(pl.LightningModule):

    def __init__(self, model, bank_lenth=128):
        super(rep_heads, self).__init__()


        if model == 'resnet18':
            self.net = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, bank_lenth)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, bank_lenth)
            )

    def forward(self, input):
        feature_representation = F.normalize(self.net(input))
        return feature_representation
