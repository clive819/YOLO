from config import COCOConfig as cc
from pycocotools.coco import COCO
import numpy as np


# MARK: - hyperparameters
annFile = './instances_train2017.json'
numAnchors = 5
maxIterations = 100


# MARK: - load data
coco = COCO(annFile)
ids = list(sorted(coco.imgs.keys()))

data = []

for idx in ids:
    annIDs = coco.getAnnIds(imgIds=idx)
    anns = coco.loadAnns(annIDs)

    imgInfo = coco.loadImgs(idx)[0]
    imgHeight, imgWidth = imgInfo['height'], imgInfo['width']

    for ann in anns:
        _, _, w, h = ann['bbox']
        w, h = w / imgWidth, h / imgHeight
        data.append([w, h])

data = np.array(data)


# MARK: - K-means
def IOU(wh, center):
    width, height = wh
    cw, ch = center

    intersectW = min(width, cw)
    intersectH = min(height, ch)

    intersectArea = intersectW * intersectH
    unionArea = width * height + cw * ch

    return intersectArea / unionArea


idx = np.random.choice(len(data), numAnchors)
centroids = sorted(data[idx], key=lambda x: np.sum(x))
oldClusters, prevDistance = None, np.inf

for iteration in range(maxIterations):
    # assign to clusters
    clusters = [[] for _ in range(numAnchors)]
    distance = []
    for p in data:
        idx, iou, d = 0, 0, 0
        for i in range(numAnchors):
            tmp = IOU(p, centroids[i])
            if tmp > iou:
                idx = i
                iou = tmp
                d = 1 - tmp
        clusters[idx].append(p)
        distance.append(d)
    clusters = np.array(clusters)
    distance = np.sum(distance)

    if distance > prevDistance or (oldClusters == clusters).all():
        break

    print('[+] iteration {}, distance: {:.8f}'.format(iteration + 1, distance))
    prevDistance = distance
    oldClusters = clusters

    # update centroids
    for i in range(numAnchors):
        z = list(zip(*clusters[i]))
        for j in range(len(centroids[i])):
            centroids[i][j] = np.mean(z[j])



# print anchors
s = '['
for c in centroids:
    s += '[{:.0f}., {:.0f}.], '.format(c[0] * cc.targetWidth, c[1] * cc.targetHeight)

print('[*] Anchors: ', end='')
print(s[:-2] + ']')
