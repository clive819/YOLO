import torchvision
import torch


torch.manual_seed(1588390)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNetConfig(object):
    # number of groups for group normalization
    numGroups = 8

    compressionRate = .5

    growthRate = 32

    # number of conv blocks
    numBlocks = [4, 4, 4, 4]


class YOLOConfig(object):
    anchors = [[5., 9.], [16., 15.], [22., 39.], [56., 77.], [142., 148.]]

    numAnchorBoxes = len(anchors)

    numClasses = 80

    # MARK: - loss
    objScale = 5.
    noObjScale = 1.
    coordScale = 1.
    classScale = 1.


class COCOConfig(object):
    targetHeight = 224
    targetWidth = 224

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((targetHeight, targetWidth)),
        torchvision.transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.25),
        torchvision.transforms.ToTensor()
    ])
