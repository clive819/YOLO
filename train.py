from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import YOLOLoss, COCODataset
from torch.optim import Adam
from config import device
from model import YOLO
import torch
import numpy as np


# MARK: - load data
cocoDataset = COCODataset('.', './instances_train2017.json', fromInternet=True)
dataLoader = DataLoader(cocoDataset, batch_size=16, shuffle=True)


# MARK: - train
model = YOLO().to(device)
criterion = YOLOLoss(warmUpEpochs=1)
optimizer = Adam(model.parameters(), lr=1e-4)
writer = SummaryWriter('./logs')
prevBestLoss = np.inf
batches = len(dataLoader)


for epoch in range(1000):
    losses = []
    for batch, (x, y) in enumerate(dataLoader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        x = model(x)
        loss = criterion(x, y, epoch)
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().item())
        print('\rEpoch {}, {} / {}'.format(epoch, batch, batches), end='')

    avgLoss = np.mean(losses)
    writer.add_scalar('Loss', avgLoss, epoch)
    print('Epoch {}, loss: {:.8f}'.format(epoch, avgLoss))

    if avgLoss < prevBestLoss:
        print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
        torch.save(model, 'model.pt')
        prevBestLoss = avgLoss
        writer.add_scalar('Model', avgLoss, epoch)

    writer.flush()
writer.close()



