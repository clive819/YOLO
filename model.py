from torchvision.ops import nms
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.warmUpBatch = 0
        self.metrics = {}

        self.backbone = DenseNet()
        self.yolo = nn.Conv2d(self.backbone.outChannels, ylc.numAnchors * (5 + ylc.numClasses), 1)

        self.apply(self.initBias)

    @staticmethod
    def initBias(module):
        if isinstance(module, nn.Conv2d):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def computeLosses(self, x, y):
        y, boxes, grid, anchors, gridDenominator, imgDenominator = y

        # MARK: - adjust prediction
        predConf = x[..., :1]
        predXY = x[..., 1:3]
        predWH = x[..., 3:5]
        predClasses = x[..., 5:]

        # MARK: - adjust ground truth
        trueConf = y[..., :1]
        trueXY = y[..., 1:3]
        trueWH = y[..., 3:5]
        trueClasses = y[..., 5:]

        # MARK: - ignore box that overlap some ground truth by 0.5
        trueBoxes = boxes
        trueBoxesXY = trueBoxes[..., :2] / gridDenominator
        trueBoxesWH = trueBoxes[..., 2:] / imgDenominator

        predBoxesXY = torch.unsqueeze(predXY, -2)
        predBoxesWH = torch.unsqueeze(predWH, -2)

        # calc IoU
        trueHalf = trueBoxesWH / 2
        trueMin = trueBoxesXY - trueHalf
        trueMax = trueBoxesXY + trueHalf

        predHalf = predBoxesWH / 2
        predMin = predBoxesXY - predHalf
        predMax = predBoxesXY + predHalf

        intersectMin = torch.max(predMin, trueMin)
        intersectMax = torch.min(predMax, trueMax)
        intersectWH = torch.max(intersectMax - intersectMin, torch.zeros_like(intersectMax))

        trueArea = trueBoxesWH[..., 0] * trueBoxesWH[..., 1]
        predArea = predBoxesWH[..., 0] * predBoxesWH[..., 1]
        intersectArea = intersectWH[..., 0] * intersectWH[..., 1]
        unionArea = trueArea + predArea - intersectArea

        iou = intersectArea / unionArea

        (bestIou, _) = torch.max(iou, -1)

        objMask = torch.unsqueeze(y[..., 0], -1)
        noObjMask = torch.unsqueeze(bestIou < .5, -1)
        coordMask = objMask

        # MARK: - compute IoU & recall
        trueHalf = trueWH / 2
        trueMin = trueXY - trueHalf
        trueMax = trueXY + trueHalf

        predHalf = predWH / 2
        predMin = predXY - predHalf
        predMax = predXY + predHalf

        intersectMin = torch.max(predMin, trueMin)
        intersectMax = torch.min(predMax, trueMax)
        intersectWH = torch.max(intersectMax - intersectMin, torch.zeros_like(intersectMax))

        trueArea = trueWH[..., 0] * trueWH[..., 1]
        predArea = predWH[..., 0] * predWH[..., 1]
        intersectArea = intersectWH[..., 0] * intersectWH[..., 1]
        unionArea = trueArea + predArea - intersectArea

        iou = intersectArea / unionArea
        iou = objMask * torch.unsqueeze(iou, -1)

        objCount = torch.sum(objMask) + 1e-7
        detectMask = (predConf * objMask >= .5).type_as(objCount)
        classMask = torch.argmax(predClasses, -1) == torch.argmax(trueClasses, -1)
        classMask = torch.unsqueeze(classMask, -1).type_as(objCount)

        recall50 = torch.sum((iou >= .50) * classMask * detectMask) / objCount
        recall75 = torch.sum((iou >= .75) * classMask * detectMask) / objCount
        avgIou = torch.sum(iou) / objCount

        # increase the loss scale for small box
        coordScale = torch.expm1(trueWH) * anchors / imgDenominator
        coordScale = torch.unsqueeze(2. - (coordScale[..., 0] * coordScale[..., 1]), -1)

        # MARK: - warm up training
        if self.warmUpBatch < tc.warmUpBatches:
            trueXY += (torch.ones_like(objMask) - objMask) * (grid + .5)
            coordMask = torch.ones_like(coordMask)
            self.warmUpBatch += 1

        # MARK: - calc total loss
        coordCount = torch.sum(coordMask) + 1e-7

        lossConf = (predConf - trueConf) * objMask * ylc.objScale + (predConf - 0) * noObjMask * ylc.noObjScale
        lossConf = torch.sum(lossConf ** 2) / (objCount + torch.sum(noObjMask))

        lossXY = coordMask * (predXY - trueXY) * coordScale * ylc.coordScale
        lossXY = torch.sum(lossXY ** 2) / coordCount

        lossWH = coordMask * (predWH - trueWH) * coordScale * ylc.coordScale
        lossWH = torch.sum(lossWH ** 2) / coordCount

        # lossClass = objMask * F.binary_cross_entropy(predClasses, trueClasses, reduction='none')
        lossClass = objMask * (predClasses - trueClasses)
        lossClass = torch.sum(lossClass ** 2) / objCount

        metrics = {
            'lossConf': lossConf,
            'lossXY': lossXY,
            'lossWH': lossWH,
            'lossClass': lossClass,
            'recall50': recall50,
            'recall75': recall75,
            'avgIou': avgIou
        }

        for key in metrics:
            metrics[key] = metrics[key].cpu().item()

        self.metrics.update(metrics)

        return lossConf + lossXY + lossWH + lossClass


    def forward(self, x, target=None, boxes=None):
        (_, _, imgH, imgW) = x.shape

        out = self.backbone(x)
        out = self.yolo(out)

        (N, _, H, W) = out.shape
        out = out.view(N, H, W, ylc.numAnchors, 5 + ylc.numClasses)

        gridX, gridY = torch.arange(W), torch.arange(H)
        gridY, gridX = torch.meshgrid(gridY, gridX)
        grid = torch.stack((gridX, gridY), -1).type_as(x).to(device)
        grid = grid.view(1, H, W, 1, 2).repeat(N, 1, 1, ylc.numAnchors, 1)

        anchors = torch.Tensor(ylc.anchors).type_as(x).view(1, 1, 1, ylc.numAnchors, 2).to(device)
        gridDenominator = torch.Tensor([W, H]).type_as(x).view(1, 1, 1, 1, 2).to(device)
        imgDenominator = torch.Tensor([imgW, imgH]).type_as(x).view(1, 1, 1, 1, 2).to(device)

        conf = torch.sigmoid(out[..., :1])
        xy = (torch.sigmoid(out[..., 1:3]) + grid) / gridDenominator
        wh = torch.expm1(out[..., 3:5]) * anchors / imgDenominator
        classes = torch.sigmoid(out[..., 5:])

        if target is not None:
            out = torch.cat([conf, xy, wh, classes], -1)
            return self.computeLosses(out, [target, boxes, grid, anchors, gridDenominator, imgDenominator])

        # MARK:- inference
        ans = []

        for confidence, coordMin, side, cat in zip(conf, xy, wh, classes):
            res = []

            # MARK: - ignore objects that has low confidence
            objMask = torch.squeeze(confidence > ylc.objThreshold, -1)
            if torch.sum(objMask) == 0:
                ans.append([])
                continue

            confidence = confidence[objMask]
            coordMin = coordMin[objMask]
            side = side[objMask]
            cat = cat[objMask]

            coordMax = coordMin + side / 2
            boxes = torch.cat((coordMin, coordMax), -1)
            categories = torch.argmax(cat, -1).type_as(boxes)

            for catID in range(ylc.numClasses):
                catIds = torch.squeeze(categories == catID, -1)

                if torch.sum(catIds) == 0:
                    continue

                score = confidence[catIds]
                box = boxes[catIds]
                category = cat[catIds]

                ids = nms(box, score, ylc.nmsThreshold)

                score = score[ids]
                box = box[ids]
                category = category[ids]

                res.append(torch.cat((score, box, category), -1))

            if res:
                ans.append(torch.cat(res, 0).tolist())
            else:
                ans.append([])

        return ans
