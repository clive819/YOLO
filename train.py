from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, Adam
from config import device, tc
from model import YOLO
from utils import *
import torch
import numpy as np


# MARK: - load data
cocoDataset = COCODataset(tc.imageDir, tc.annFile, fromInternet=False if tc.imageDir else True)
dataLoader = DataLoader(cocoDataset, batch_size=tc.batchSize, shuffle=True)


# MARK: - train
model = YOLO().to(device)
if tc.preTrainedWeight:
    model.load_state_dict(torch.load(tc.preTrainedWeight, map_location=device))
    model.warmUpBatch = tc.warmUpBatches

optimizer = Adam(model.parameters(), lr=1e-4)
prevBestLoss = np.inf
batches = len(dataLoader)
logger = MetricsLogger()


model.train()
for epoch in range(tc.epochs):
    losses = []
    for batch, (x, y, z) in enumerate(dataLoader):
        x, y, z = x.to(device), y.to(device), z.to(device)

        loss = model(x, y, z)
        losses.append(loss.cpu().item())

        metrics = model.metrics
        logger.step(metrics, epoch, batch)
        logger.step({'Loss': losses[-1]}, epoch, batch)
        print('Epoch {} | {} / {}'.format(epoch, batch, batches), end='')
        for key in metrics:
            print(' | {}: {:.4f}'.format(key, metrics[key]), end='')
        print(' | loss: {:.4f}\r'.format(losses[-1]), end='')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.epochEnd(epoch)
    avgLoss = np.mean(losses)
    print('\nEpoch {}, loss: {:.8f}'.format(epoch, avgLoss))

    if avgLoss < prevBestLoss:
        print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
        torch.save(model.state_dict(), 'model.pt')
        prevBestLoss = avgLoss
        logger.addScalar('Model', avgLoss, epoch)
    logger.flush()
logger.close()
