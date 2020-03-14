from config import DenseNetConfig as dnc
from config import YOLOConfig as ylc
from config import device
import torch
import torch.nn as nn


class _transitionLayer(nn.Module):
    def __init__(self, inChannels):
        super(_transitionLayer, self).__init__()

        self.outChannels = int(inChannels * dnc.compressionRate)

        self.module = nn.Sequential(
            nn.GroupNorm(dnc.numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, self.outChannels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.module(x)


class _convBlock(nn.Module):
    def __init__(self, inChannels):
        super(_convBlock, self).__init__()

        self.outChannels = dnc.growthRate
        self.module = nn.Sequential(
            nn.GroupNorm(dnc.numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, 4 * dnc.growthRate, 1),
            nn.GroupNorm(dnc.numGroups, 4 * dnc.growthRate),
            nn.ReLU(),
            nn.Conv2d(4 * dnc.growthRate, dnc.growthRate, 3, padding=1)
        )

    def forward(self, x):
        return self.module(x)


class _denseBlock(nn.Module):
    def __init__(self, inChannels, numBlocks):
        super(_denseBlock, self).__init__()

        self.outChannels = inChannels

        self.layers = nn.ModuleList()
        for _ in range(numBlocks):
            self.layers.append(_convBlock(self.outChannels))
            self.outChannels += dnc.growthRate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))

        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.outChannels = 64

        self.input = nn.Sequential(
            nn.Conv2d(3, self.outChannels, 7, padding=3),
            nn.GroupNorm(dnc.numGroups, self.outChannels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        layers = [self.input]

        for num in dnc.numBlocks:
            block = _denseBlock(self.outChannels, num)
            self.outChannels = block.outChannels
            trans = _transitionLayer(self.outChannels)
            self.outChannels = trans.outChannels
            layers.append(block)
            layers.append(trans)

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()

        self.backbone = DenseNet()

        self.yolo = nn.Conv2d(self.backbone.outChannels, ylc.numAnchorBoxes * (5 + ylc.numClasses), 1)

    def forward(self, x):
        (_, _, imgH, imgW) = x.shape

        out = self.backbone(x)
        out = self.yolo(out)

        (N, _, H, W) = out.shape
        out = out.view(N, H, W, ylc.numAnchorBoxes, 5 + ylc.numClasses)

        gridX = torch.arange(W).type_as(x).repeat(H, 1).reshape(1, H, W, 1, 1).to(device)
        gridY = gridX.transpose(1, 2)
        grid = torch.cat((gridX, gridY), -1).repeat(N, 1, 1, ylc.numAnchorBoxes, 1)

        anchors = torch.Tensor(ylc.anchors).type_as(x).reshape(1, 1, 1, ylc.numAnchorBoxes, 2).to(device)
        gridDenominator = torch.Tensor([W, H]).type_as(x).reshape(1, 1, 1, 1, 2)
        imgDenominator = torch.Tensor([imgW, imgH]).type_as(x).reshape(1, 1, 1, 1, 2)

        conf = torch.sigmoid(out[..., :1])
        xy = (torch.sigmoid(out[..., 1:3]) + grid) / gridDenominator
        wh = torch.exp(out[..., 3:5]) * anchors / imgDenominator
        classes = torch.sigmoid(out[..., 5:])

        return torch.cat([conf, xy, wh, classes], -1)
