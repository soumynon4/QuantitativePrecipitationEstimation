# @Time     : 1/8/2022 1:22 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : MLP_OneEpoch.py
# @Project  : QuantitativePrecipitationEstimation
import sys

import torch
from torch.utils.data import DataLoader


def trainOneEpoch(name: str,
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  dataLoader: DataLoader,
                  device: torch.device,
                  epoch: int,
                  lossFunction: torch.nn.Module):
    model.train()
    optimizer.zero_grad()

    accumulatedLoss = torch.zeros(1).to(device)  # 累计损失

    totalSteps = len(dataLoader)

    for step, data in enumerate(dataLoader):
        _, _, features, labels = data
        features = features[:, :, 4, 4]
        prediction = model(features.to(device))

        loss = lossFunction(prediction, labels.to(device))
        loss.backward()
        accumulatedLoss += loss.detach()

        print("[Model]: {} | [State]: training | [Epoch]: {} | [Step]: {} | [Loss]: {:.3f}".format(
            name,
            epoch,
            step + 1,
            loss.item()
        ))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accumulatedLoss.item() / totalSteps


@torch.no_grad()
def evaluateOneEpoch(name: str,
                     model: torch.nn.Module,
                     dataLoader: DataLoader,
                     device: torch.device,
                     epoch: int,
                     lossFunction: torch.nn.Module):
    model.eval()

    accumulatedLoss = torch.zeros(1).to(device)  # 累计损失

    totalSteps = len(dataLoader)

    for step, data in enumerate(dataLoader):
        _, _, features, labels = data
        features = features[:, :, 4, 4]
        prediction = model(features.to(device))

        loss = lossFunction(prediction, labels.to(device))
        accumulatedLoss += loss

        print("[Model]: {} | [State]: training(validating) | [Epoch]: {} | [Step]: {} | [Loss]: {:.3f}".format(
            name,
            epoch,
            step + 1,
            loss.item())
        )

    return accumulatedLoss.item() / totalSteps