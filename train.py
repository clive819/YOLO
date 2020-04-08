from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import COCODataset
from torch.optim import SGD, Adam
from config import device, tc
from model import YOLO
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
writer = SummaryWriter('./logs')
prevBestLoss = np.inf
batches = len(dataLoader)


model.train()
for epoch in range(tc.epochs):
    losses = []
    for batch, (x, y, z) in enumerate(dataLoader):
        x, y, z = x.to(device), y.to(device), z.to(device)

        loss = model(x, y, z)
        losses.append(loss.cpu().item())

        metrics = model.metrics
        print('Epoch {} | {} / {}'.format(epoch, batch, batches), end='')
        for key in metrics:
            print(' | {}: {:.4f}'.format(key, metrics[key]), end='')
        print(' | loss: {:.4f}\r'.format(losses[-1]), end='')

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        for key in metrics:
            writer.add_scalar('Epoch {}/{}'.format(epoch, key), metrics[key], batch)

    avgLoss = np.mean(losses)
    writer.add_scalar('Avg loss', avgLoss, epoch)
    print('Epoch {}, loss: {:.8f}'.format(epoch, avgLoss))

    if avgLoss < prevBestLoss:
        print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
        torch.save(model.state_dict(), 'model.pt')
        prevBestLoss = avgLoss
        writer.add_scalar('Model', avgLoss, epoch)
    writer.flush()
writer.close()
