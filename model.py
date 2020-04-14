from torchvision.ops import nms
from config import *
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

        self.warmUpBatch = 0
        self.metrics = {}

        self.backbone = DenseNet()
        self.yolo = nn.Conv2d(self.backbone.outChannels, ylc.numAnchors * (5 + ylc.numClasses), 1)

    def computeLosses(self, x, y):
        y, boxes, grid, anchors, gridSize, imgSize = y

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
        trueBoxesXY = trueBoxes[..., :2] / gridSize
        trueBoxesWH = trueBoxes[..., 2:] / imgSize

        predBoxesXY = torch.unsqueeze(predXY / gridSize, -2)
        predBoxesWH = torch.unsqueeze(torch.expm1(predWH) * anchors / imgSize, -2)

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
        tXY = trueXY / gridSize
        tWH = torch.expm1(trueWH) * anchors / imgSize
        pXY = predXY / gridSize
        pWH = torch.expm1(predWH) * anchors / imgSize

        trueHalf = tWH / 2
        trueMin = tXY - trueHalf
        trueMax = tXY + trueHalf

        predHalf = pWH / 2
        predMin = pXY - predHalf
        predMax = pXY + predHalf

        intersectMin = torch.max(predMin, trueMin)
        intersectMax = torch.min(predMax, trueMax)
        intersectWH = torch.max(intersectMax - intersectMin, torch.zeros_like(intersectMax))

        trueArea = tWH[..., 0] * tWH[..., 1]
        predArea = pWH[..., 0] * pWH[..., 1]
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
        coordScale = torch.expm1(trueWH) * anchors / imgSize
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

        lossClass = nn.functional.binary_cross_entropy_with_logits(predClasses * objMask, trueClasses * objMask)

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
        grid = torch.stack((gridX, gridY), -1).type_as(x)
        grid = grid.view(1, H, W, 1, 2).repeat(N, 1, 1, ylc.numAnchors, 1)

        anchors = torch.Tensor(ylc.anchors).type_as(x).view(1, 1, 1, ylc.numAnchors, 2)
        gridSize = torch.Tensor([W, H]).type_as(x).view(1, 1, 1, 1, 2)
        imgSize = torch.Tensor([imgW, imgH]).type_as(x).view(1, 1, 1, 1, 2)

        conf = torch.sigmoid(out[..., :1])
        xy = torch.sigmoid(out[..., 1:3]) + grid
        wh = out[..., 3:5]
        classes = out[..., 5:]

        if target is not None:
            out = torch.cat([conf, xy, wh, classes], -1)
            return self.computeLosses(out, [target, boxes, grid, anchors, gridSize, imgSize])

        # MARK:- inference
        ans = []

        xy /= gridSize
        wh = torch.expm1(out[..., 3:5]) * anchors / imgSize

        for confidence, coord, side, cat in zip(conf, xy, wh, classes):
            res = []

            # MARK: - ignore objects that has low confidence
            objMask = torch.squeeze(confidence > ylc.objThreshold, -1)
            if torch.sum(objMask) == 0:
                ans.append([])
                continue

            confidence = confidence[objMask]
            coord = coord[objMask]
            side = side[objMask]
            cat = cat[objMask]

            x1y1 = coord - side / 2
            x2y2 = coord + side / 2
            boxes = torch.cat((x1y1, x2y2), -1)
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
