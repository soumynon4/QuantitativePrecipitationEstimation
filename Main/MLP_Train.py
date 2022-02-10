# @Time     : 1/8/2022 1:07 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : MLP_Train.py
# @Project  : QuantitativePrecipitationEstimation
import os
import math
import logging

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from Kit.MLP_OneEpoch import trainOneEpoch, evaluateOneEpoch
from Kit.RadarGaugeDataset import DatasetForQuantitativePrecipitationEstimation
from Kit.CustomizedLoss import Loss
from Kit.Tools import set_logger

from Methods.MLP import MultilayerPerceptron


def main(modelName: str,
         learningRate: float = 0.001,
         learningRateFactor: float = 0.01,
         batchSize: int = 10,
         epochs: int = 100,
         weightFolder: str = "/media/data3/zhangjianchang/DATA/ModelCheckPoints/MLP",
         logFilePath: str = "/media/data3/zhangjianchang/DATA/Log/MLP.log"):
    # weight folder change 2
    # log file change 3

    set_logger(logFilePath)

    runningDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("[Model]: {} | [State]: training".format(modelName))
    logging.info("[Model]: {} | [State]: training | [Learning Rate]: {} | [Batch Size]: {} | [Epoch Number]: {}".format(
        modelName,
        learningRate,
        batchSize,
        epochs
    ))

    if os.path.exists(weightFolder) is False:
        logging.info("[Model]: {} | [State]: training | [Weight Folder]: doesn't exist".format(modelName))
        os.makedirs(weightFolder)

    logging.info("[Model]: {} | [State]: training | [Dataset]: loading".format(modelName))

    # Dataset Creating
    trainDataset = DatasetForQuantitativePrecipitationEstimation(
        "/media/data3/zhangjianchang/DATA/Dataset/CSV/TrainDataset.csv",
        "/media/data3/zhangjianchang/DATA/Dataset/RAIN",
        isNormalize=True
    )
    validateDataset = DatasetForQuantitativePrecipitationEstimation(
        "/media/data3/zhangjianchang/DATA/Dataset/CSV/ValidateDataset.csv",
        "/media/data3/zhangjianchang/DATA/Dataset/RAIN",
        isNormalize=True
    )

    nw = min([os.cpu_count(), 8])  # number of workers
    logging.info("[Model]: {} | [State]: training | [Dataloader Workers]: {}".format(modelName, nw))
    logging.info("[Model]: {} | [State]: training | [Dataset]: finish".format(modelName))

    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers=nw,
                                              drop_last=True)

    validateLoader = torch.utils.data.DataLoader(dataset=validateDataset,
                                                 batch_size=batchSize,
                                                 shuffle=False,
                                                 num_workers=nw,
                                                 drop_last=True)

    # model change 4
    model = MultilayerPerceptron(inputFeatureSize=4).to(runningDevice)

    optimizer = optim.AdamW(model.parameters(), lr=learningRate)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - learningRateFactor) + learningRateFactor)

    # Loss function definition
    trainLossFunction = Loss().to(runningDevice)
    validateLossFunction = torch.nn.L1Loss()

    for epoch in range(epochs):
        if epoch >= epochs // 3:
            trainLossFunction = torch.nn.HuberLoss(delta=10)
        thisTrainEpochMeanLoss = trainOneEpoch(name=modelName, model=model, optimizer=optimizer,
                                               dataLoader=trainLoader, device=runningDevice,
                                               epoch=epoch, lossFunction=trainLossFunction)

        logging.info("[Model]: {} | [State]: training | [Train Dataset Mean Loss]: {} | [Epoch]: {}".format(
            modelName,
            thisTrainEpochMeanLoss,
            epoch
        ))

        thisValidateEpochMeanLoss = evaluateOneEpoch(name=modelName, model=model, dataLoader=validateLoader,
                                                     device=runningDevice, epoch=epoch,
                                                     lossFunction=validateLossFunction)

        logging.info(
            "[Model]: {} | [State]: training(validating) | [Validate Dataset Mean Loss]: {} | [Epoch]: {}".format(
                modelName,
                thisValidateEpochMeanLoss,
                epoch
            ))

        torch.save(model.state_dict(), os.path.join(weightFolder, "{}-{}.pth").format(modelName, epoch))

        logging.info("[Model]: {} | [State]: training | [Epoch]: {} | [Learning Rate] : {}".format(
            modelName,
            epoch,
            optimizer.param_groups[0]["lr"]
        ))

        scheduler.step()

    logging.info("[Model]: {} | [State]: finish training".format(modelName))


if __name__ == '__main__':
    # change1
    main("MLP", learningRate=0.0008, batchSize=256, epochs=20)
