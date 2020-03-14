from torchsummary import summary
from model import YOLO


model = YOLO()
summary(model, (3, 224, 224))
