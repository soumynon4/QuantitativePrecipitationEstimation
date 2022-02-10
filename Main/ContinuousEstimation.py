# @Time     : 2/7/2022 3:46 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ContinuousEstimation.py
# @Project  : QuantitativePrecipitationEstimation
import os
import math
import logging
import pickle

import numpy

import torch
from torch.utils.data import DataLoader

from Kit.RadarGaugeDataset import ContinuousDatasetForQuantitativePrecipitationEstimation

from Methods.ViT import VisionTransformer
from Methods.ShuffleNet import shufflenet_v2_x1_0
from Methods.ResNet import resnet101
from Methods.MobileNetV3 import mobilenet_v3_large
from Methods.EfficientNetV2 import efficientnetv2_l
from Methods.DenseNet import densenet121
from Methods.QPET import QuantitativePrecipitationEstimationTransformer

from Kit.Tools import set_logger


def main(modelName: str,
         modelWeightPath: str,
         batchSize: int,
         lossFunction: torch.nn.Module,
         logFilePath: str = "/media/data3/zhangjianchang/DATA",
         isSave: bool = False,
         saveFolder: str = "/media/data3/zhangjianchang/DATA"
         ):
    # weight folder change 3
    # log file change 4
    assert batchSize == 1

    set_logger(logFilePath)
    runningDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("[Model]: {} | [State]: testing".format(modelName))

    # Dataset Creating
    testDataset = ContinuousDatasetForQuantitativePrecipitationEstimation(
        "c:\\users\\admin\\downloads\\csv\\SFL0309_20210705.csv",
        "c:\\users\\admin\\downloads\\rain",
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

    # model change 5

    # model = VisionTransformer(img_size=9, patch_size=3, in_c=4, num_classes=1).to(runningDevice)
    # model = shufflenet_v2_x1_0(num_classes=1).to(runningDevice)
    # model = resnet101(num_classes=1).to(runningDevice)
    # model = mobilenet_v3_large(num_classes=1).to(runningDevice)
    # model = efficientnetv2_l(num_classes=1).to(runningDevice)
    # model = densenet121(num_classes=1).to(runningDevice)
    model = QuantitativePrecipitationEstimationTransformer().to(runningDevice)

    model.load_state_dict(torch.load(modelWeightPath, map_location=runningDevice))
    model.eval()

    with torch.no_grad():
        accumulatedLoss = torch.zeros(1).to(runningDevice)

        totalSteps = len(testLoader)

        for step, data in enumerate(testLoader):
            names, seconds, features, labels = data

            if seconds[0] == 0:
                GroundTruthLists.append(numpy.zeros((1, 1)))
                EstimationLists.append(numpy.zeros((1, 1)))
                FileNameLists.extend(names)
                ScanSecondsLists.extend(seconds.cpu().numpy())
                continue

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
    # change1 Name
    # change 2 weight path
    Name = "QPET"
    main(modelName=Name,
         modelWeightPath="/media/data3/zhangjianchang/DATA/ModelCheckPoints/{}/QPET-6.pth".format(Name),
         batchSize=1,
         lossFunction=torch.nn.L1Loss(),
         isSave=True)



