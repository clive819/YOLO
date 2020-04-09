from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from io import BytesIO
from PIL import Image
from config import *
import os
import torch
import requests
import numpy as np


class COCODataset(Dataset):
    def __init__(self, root, annFile, fromInternet=False):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.fromInternet = fromInternet

        self.newIdx = {}
        self.oldIdx = {}
        for i, cat in enumerate(self.coco.cats):
            self.newIdx[cat] = i
            self.oldIdx[i] = cat

    def __len__(self):
        return len(self.ids)

    def decodeClassID(self, idx):
        return self.coco.cats[self.oldIdx[idx]]['name']

    @staticmethod
    def calcIou(w1, h1, w2, h2):
        intersectW = min(w1, w2)
        intersectH = min(h1, h2)

        intersectArea = intersectW * intersectH
        unionArea = w1 * h1 + w2 * h2

        return intersectArea / unionArea


    def findBestAnchor(self, w, h):
        idx, iou = 0, 0

        for i in range(ylc.numAnchors):
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
        target = torch.zeros(gridH, gridW, ylc.numAnchors, 5 + ylc.numClasses)
        boxes, boxID = torch.zeros(1, 1, 1, cc.maxBoxPerImage, 4), 0

        for ann in anns:
            x, y, w, h = ann['bbox']
            c = self.newIdx[ann['category_id']]

            centerX = x + w / 2
            centerX = centerX / imgWidth * gridW

            centerY = y + h / 2
            centerY = centerY / imgHeight * gridH

            w = w / imgWidth * cc.targetWidth
            h = h / imgHeight * cc.targetHeight

            bestAnchorIdx = self.findBestAnchor(w, h)
            anchor = ylc.anchors[bestAnchorIdx]

            width = np.log1p(w / anchor[0])
            height = np.log1p(h / anchor[1])

            gridX = int(np.floor(centerX))
            gridY = int(np.floor(centerY))

            target[gridY, gridX, bestAnchorIdx, 0] = 1.
            target[gridY, gridX, bestAnchorIdx, 1:5] = torch.as_tensor((centerX, centerY, width, height))
            target[gridY, gridX, bestAnchorIdx, 5 + c] = 1.

            boxes[0, 0, 0, boxID] = torch.as_tensor((centerX, centerY, w, h))
            boxID = (boxID + 1) % cc.maxBoxPerImage
        boxes = torch.as_tensor(boxes)

        return cc.transforms(img), target, boxes


class MetricsLogger(object):
    def __init__(self, folder='./logs'):
        self.writer = SummaryWriter(folder)
        self.memory = {}

    def addScalar(self, tag: str, value, step=None, wallTime=None):
        self.writer.add_scalar(tag, value, step, wallTime)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def step(self, metrics: dict, epoch: int, batch: int):
        for key in metrics:
            self.writer.add_scalar('Epoch {}/{}'.format(epoch, key), metrics[key], batch)
            if key in self.memory:
                self.memory[key].append(metrics[key])
            else:
                self.memory[key] = [metrics[key]]

    def epochEnd(self, epoch: int):
        for key in self.memory:
            avg = np.mean(self.memory[key])
            self.writer.add_scalar('Average/{}'.format(key), avg, epoch)
        self.memory.clear()
