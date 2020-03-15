from torch.utils.data.dataset import Dataset
from config import DenseNetConfig as dnc
from config import YOLOConfig as ylc
from config import COCOConfig as cc
from pycocotools.coco import COCO
from config import device
from io import BytesIO
from PIL import Image
import os
import torch
import requests
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# TODO: decode output


class COCODataset(Dataset):
    def __init__(self, root, annFile, fromInternet=False):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.fromInternet = fromInternet

        self.newIdx = {}
        for i, cat in enumerate(self.coco.cats):
            self.newIdx[cat] = i

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def calcIou(w1, h1, w2, h2):
        intersectW = min(w1, w2)
        intersectH = min(h1, h2)

        intersectArea = intersectW * intersectH
        unionArea = w1 * h1 + w2 * h2

        return intersectArea / unionArea


    def findBestAnchor(self, w, h):
        idx, iou = 0, 0

        for i in range(ylc.numAnchorBoxes):
            tmp = self.calcIou(w, h, ylc.anchors[i][0], ylc.anchors[i][1])
            if tmp > iou:
                idx = i
                iou = tmp

        return idx

    def __getitem__(self, idx):
        coco = self.coco
        imgID = self.ids[idx]
        annIDs = coco.getAnnIds(imgIds=imgID)
        anns = coco.loadAnns(annIDs)

        imgInfo = coco.loadImgs(imgID)[0]
        imgHeight, imgWidth = imgInfo['height'], imgInfo['width']
        if self.fromInternet:
            imgData = BytesIO(requests.get(imgInfo['coco_url']).content)
            img = Image.open(imgData).convert('RGB')
        else:
            imgName = imgInfo['file_name']
            img = Image.open(os.path.join(self.root, imgName)).convert('RGB')

        compressionRatio = 2 ** (len(dnc.numBlocks) + 1)

        gridH, gridW = cc.targetHeight // compressionRatio, cc.targetWidth // compressionRatio

        # conf, centerX, centerY, width, height, original w, original h, classes
        target = torch.zeros(gridH, gridW, ylc.numAnchorBoxes, 5+2+ylc.numClasses)

        for ann in anns:
            x, y, w, h = ann['bbox']
            c = self.newIdx[ann['category_id']]

            centerX = x + w / 2.
            centerX = centerX / imgWidth * gridW

            centerY = y + h / 2.
            centerY = centerY / imgHeight * gridH

            w = w / imgWidth * cc.targetWidth + 1e-7
            h = h / imgHeight * cc.targetHeight + 1e-7

            bestAnchorIdx = self.findBestAnchor(w, h)
            anchor = ylc.anchors[bestAnchorIdx]

            width = max(0., np.log(w) / anchor[0])
            height = max(0., np.log(h) / anchor[1])

            gridX = int(np.floor(centerX))
            gridY = int(np.floor(centerY))

            target[gridY, gridX, bestAnchorIdx, 0] = 1.
            target[gridY, gridX, bestAnchorIdx, 1:5] = torch.Tensor([centerX, centerY, width, height])
            target[gridY, gridX, bestAnchorIdx, 5:7] = torch.Tensor([w, h])
            target[gridY, gridX, bestAnchorIdx, 7 + c] = 1.

        return cc.transforms(img), target



class YOLOLoss(nn.Module):
    def __init__(self, warmUpEpochs=1):
        super(YOLOLoss, self).__init__()

        self.warmUpEpochs = warmUpEpochs

    @staticmethod
    def extractBoundingBoxes(y):
        maxCount = 1
        tmp = []

        for grid in y:
            g = grid[grid[..., 0] == 1]
            b = torch.cat([g[..., 1:3], g[..., 5:7]], -1)
            tmp.append(b)
            maxCount = max(maxCount, len(b))

        N = len(tmp)
        boxes = torch.zeros(N, 1, 1, 1, maxCount, 4).type_as(y).to(device)

        for i in range(N):
            box = tmp[i]
            boxes[i, 0, 0, 0, :len(box)] = box

        return boxes

    def forward(self, x, y, epoch):
        (N, H, W, _, _) = x.shape

        gridX = torch.arange(W).type_as(x).repeat(H, 1).reshape(1, H, W, 1, 1).to(device)
        gridY = gridX.transpose(1, 2)
        grid = torch.cat((gridX, gridY), -1).repeat(N, 1, 1, ylc.numAnchorBoxes, 1)

        anchors = torch.Tensor(ylc.anchors).type_as(x).reshape(1, 1, 1, ylc.numAnchorBoxes, 2).to(device)
        gridDenominator = torch.Tensor([W, H]).type_as(x).reshape(1, 1, 1, 1, 2)
        imgDenominator = torch.Tensor([cc.targetWidth, cc.targetHeight]).type_as(x).reshape(1, 1, 1, 1, 2)

        # MARK: - adjust prediction
        predConf = x[..., :1]
        predXY = x[..., 1:3]
        predWH = x[..., 3:5]
        predClasses = x[..., 5:]


        # MARK: - adjust ground truth
        trueConf = y[..., :1]
        trueXY = y[..., 1:3]
        trueWH = y[..., 3:5]
        trueClasses = y[..., 7:]


        # MARK: - ignore box that overlap some ground truth by 0.5
        trueBoxes = self.extractBoundingBoxes(y)
        trueBoxesXY = trueBoxes[..., :2] / gridDenominator
        trueBoxesWH = trueBoxes[..., 2:] / imgDenominator

        predBoxesXY = torch.unsqueeze(predXY, -2)
        predBoxesWH = torch.unsqueeze(predWH, -2)

        # calc IoU
        trueHalf = trueBoxesWH / 2.
        trueMin = trueBoxesXY - trueHalf
        trueMax = trueBoxesXY + trueHalf

        predHalf = predBoxesWH / 2.
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

        # increase the loss scale for small box
        coordScale = torch.exp(trueWH) * anchors / imgDenominator
        coordScale = torch.unsqueeze(2. - (coordScale[..., 0] * coordScale[..., 1]), -1)


        # MARK: - warm up training
        if epoch < self.warmUpEpochs:
            coordMask = torch.ones_like(coordMask)
            trueXY += (1. - objMask) * (.5 + grid)


        # MARK: - calc total loss
        lossConf = (predConf - trueConf) * objMask * ylc.objScale + (predConf - 0.) * noObjMask * ylc.noObjScale
        lossConf = torch.sum(lossConf ** 2)

        lossXY = coordMask * (predXY - trueXY) * coordScale * ylc.coordScale
        lossXY = torch.sum(lossXY ** 2)

        lossWH = coordMask * (predWH - trueWH) * coordScale * ylc.coordScale
        lossWH = torch.sum(lossWH ** 2)

        lossClass = objMask * F.binary_cross_entropy(predClasses, trueClasses, reduction='none')
        lossClass = torch.sum(lossClass)

        loss = lossConf + lossXY + lossWH + lossClass

        return loss












