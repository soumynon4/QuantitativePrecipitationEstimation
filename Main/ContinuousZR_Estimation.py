# @Time     : 2/8/2022 3:49 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ContinuousZR_Estimation.py
# @Project  : QuantitativePrecipitationEstimation
import os
import math
import logging
import pickle

import numpy

import torch
from torch.utils.data import DataLoader

from Kit.RadarGaugeDataset import ContinuousDatasetForQuantitativePrecipitationEstimation
from Methods.ReflectivityRainrateRelation import QPEbyZh, QPEbyZhAndZdr
from Kit.Tools import set_logger


def main(modelName: str,
         batchSize: int,
         lossFunction: torch.nn.Module,
         logFilePath: str = "/media/data3/zhangjianchang/DATA/Log/ZRZDR.log",
         isSave: bool = False,
         saveFolder: str = "/media/data3/zhangjianchang/DATA/Estimation/ZRZDR"
         ):
    # change 2 log
    # change 3 save Folder
    assert batchSize == 1

    set_logger(logFilePath)
    logging.info("[Model]: {} | [State]: testing".format(modelName))

    # Dataset Creating
    testDataset = ContinuousDatasetForQuantitativePrecipitationEstimation(
        "/media/data3/zhangjianchang/DATA/Dataset/CSV/TestDataset.csv",
        "/media/data3/zhangjianchang/DATA/Dataset/RAIN",
        isNormalize=False
    )

    GroundTruthLists = []
    EstimationLists = []
    FileNameLists = []
    ScanSecondsLists = []

    nw = min([os.cpu_count(), 8])  # number of workers

    testLoader = torch.utils.data.DataLoader(testDataset,
                                             batch_size=batchSize,
                                             shuffle=False,
                                             num_workers=nw)

    accumulatedLoss = torch.zeros(1)
    totalSteps = len(testLoader)

    for step, data in enumerate(testLoader):
        names, seconds, features, labels = data

        if seconds[0] == 0:
            GroundTruthLists.append(numpy.zeros((1, 1)))
            EstimationLists.append(numpy.zeros((1, 1)))
            FileNameLists.extend(names)
            ScanSecondsLists.extend(seconds.cpu().numpy())
            continue

        dBZ = features.cpu().numpy()[:, 0, 4, 4]
        Zdr = features.cpu().numpy()[:, 2, 4, 4]

        # change 4 model
        # prediction = QPEbyZh(dBZ=dBZ, coefficient=200, index=1.6)
        # prediction = QPEbyZh(dBZ=dBZ, coefficient=250, index=1.2)
        # prediction = QPEbyZh(dBZ=dBZ, coefficient=300, index=1.4)
        prediction = QPEbyZhAndZdr(dBZ=dBZ, Zdr=Zdr)
        prediction = torch.unsqueeze(torch.from_numpy(prediction), dim=1)

        loss = lossFunction(prediction, labels.cpu())
        accumulatedLoss += loss

        GroundTruthLists.append(labels.cpu().numpy())
        EstimationLists.append(prediction.cpu().numpy())
        FileNameLists.extend(names)
        ScanSecondsLists.extend(seconds.cpu().numpy())

        print("[Model]: {} | [State]: testing | [Step]: {}, [Loss]: {:.3f}".format(
            modelName,
            step + 1,
            loss.item()
        ))

    logging.info("[Model]: {} | [State]: testing | [Mean Loss]: {:.3f}".format(
        modelName,
        accumulatedLoss.item() / totalSteps
    ))
    logging.info("[Model]: {} | [State]: finish testing".format(modelName))

    GroundTruthNumpy = numpy.concatenate(GroundTruthLists, axis=0)
    EstimationNumpy = numpy.concatenate(EstimationLists, axis=0)
    if isSave:
        numpy.save(os.path.join(saveFolder, "{}_Test_GroundTruth_RainRate".format(modelName)), GroundTruthNumpy)

        numpy.save(os.path.join(saveFolder, "{}_Test_Estimation_RainRate".format(modelName)), EstimationNumpy)

        with open(os.path.join(saveFolder, "{}_Test_Filenames.pkl".format(modelName)), "wb") as handle1:
            pickle.dump(FileNameLists, handle1)

        with open(os.path.join(saveFolder, "{}_Test_Scan_Seconds.pkl".format(modelName)), "wb") as handle2:
            pickle.dump(ScanSecondsLists, handle2)
        print("Finish Saving.")

    return accumulatedLoss.item() / totalSteps


if __name__ == '__main__':
    # change 1
    main("ZRZDR", batchSize=256, lossFunction=torch.nn.L1Loss(), isSave=True)