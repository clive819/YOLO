import torchvision
import torch


torch.manual_seed(1588390)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNetConfig(object):
    # number of groups for group normalization
    numGroups       = 8
    growthRate      = 32
    compressionRate = .5

    # number of conv blocks
    numBlocks = [4, 4, 4, 4]


class YOLOConfig(object):
    numClasses  = 80
    anchors     = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
    numAnchors  = len(anchors)

    # MARK: - loss
    objScale    = 5.
    noObjScale  = 1.
    coordScale  = 1.
    classScale  = 1.

    # MARK: - inference
    objThreshold = .50
    nmsThreshold = .45


class COCOConfig(object):
    maxBoxPerImage  = 93
    targetHeight    = 416
    targetWidth     = 416

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((targetHeight, targetWidth)),
        torchvision.transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.125),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class TrainingConfig(object):
    epochs          = 100
    batchSize       = 8
    warmUpBatches   = 10000

    annFile             = '/kaggle/input/coco2017/annotations_trainval2017/annotations/instances_train2017.json'
    imageDir            = '/kaggle/input/coco2017/train2017/train2017'
    preTrainedWeight    = ''


dnc = DenseNetConfig
ylc = YOLOConfig
cc  = COCOConfig
tc = TrainingConfig
