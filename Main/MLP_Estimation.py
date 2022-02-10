# @Time     : 1/8/2022 1:07 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : MLP_Estimation.py
# @Project  : QuantitativePrecipitationEstimation
import os
import math
import logging
import pickle

import numpy

import torch
from torch.utils.data import DataLoader

from Kit.RadarGaugeDataset import DatasetForQuantitativePrecipitationEstimation
from Methods.MLP import MultilayerPerceptron
from Kit.Tools import set_logger


def main(modelName: str,
         modelWeightPath: str,
         batchSize: int,
         lossFunction: torch.nn.Module,
         logFilePath: str = "/media/data3/zhangjianchang/DATA/Log/MLP.log",
         isSave: bool = False,
         saveFolder: str = "/media/data3/zhangjianchang/DATA/Estimation/MLP"
         ):
    set_logger(logFilePath)
    runningDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("[Model]: {} | [State]: testing".format(modelName))

    # Dataset Creating
    testDataset = DatasetForQuantitativePrecipitationEstimation(
        "/media/data3/zhangjianchang/DATA/Dataset/CSV/TestDataset.csv",
        "/media/data3/zhangjianchang/DATA/Dataset/RAIN",
        isNormalize=True
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

    model = MultilayerPerceptron(inputFeatureSize=4).to(runningDevice)
    model.load_state_dict(torch.load(modelWeightPath, map_location=runningDevice))
    model.eval()

    with torch.no_grad():
        accumulatedLoss = torch.zeros(1).to(runningDevice)

        totalSteps = len(testLoader)

        for step, data in enumerate(testLoader):
            names, seconds, features, labels = data
            features = features[:, :, 4, 4]
            prediction = model(features.to(runningDevice))

            loss = lossFunction(prediction, labels.to(runningDevice))
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
    main("MLP",
         modelWeightPath="/media/data3/zhangjianchang/DATA/ModelCheckPoints/MLP/MLP-19.pth",
         batchSize=256,
         lossFunction=torch.nn.L1Loss(),
         isSave=True)
