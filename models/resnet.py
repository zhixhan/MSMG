from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn


class Backbone(nn.Module):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.res2=nn.Sequential(OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                ]
            )
        )
        self.res3 = resnet.layer2  # res3
        self.res4 = resnet.layer3  # res4
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat2 = self.res2(x)
        feat3 = self.res3(feat2)
        feat4 = self.res4(feat3)
        return OrderedDict([["feat_res2", feat2], ["feat_res3", feat3], ["feat_res4", feat4]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat2 = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat2]]), feat

class RegHead(nn.Sequential):
    def __init__(self):
        resnet = torchvision.models.resnet.__dict__["resnet50"](pretrained=True)

        # freeze layers
        resnet.conv1.weight.requires_grad_(False)
        resnet.bn1.weight.requires_grad_(False)
        resnet.bn1.bias.requires_grad_(False)
        super(RegHead, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(RegHead, self).forward(x)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["location_feat_res4", feat]])

def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)

