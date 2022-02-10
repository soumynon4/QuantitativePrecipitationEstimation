# @Time     : 1/7/2022 8:07 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : TestDataset_RainRate_Rainfall_Hourly_Convert.py
# @Project  : QuantitativePrecipitationEstimation
import os
import pickle
from collections import defaultdict

import numpy


def RainRateConvertRainfallHourly(Pred: numpy.ndarray, Ground: numpy.ndarray,
                                  Filenames: list, Seconds: list,
                                  saveFolder: str, modelName: str):

    EstimationDict = defaultdict(list)
    LabelDict = defaultdict(list)

    RainfallHourlyEstimationList = []
    RainfallHourlyLabelList = []

    for i in range(len(Filenames)):
        name = Filenames[i]
        hashValue = name.split('_')[-1].split('.')[0] + '_' + name[:11]

        second = Seconds[i].item()
        label = Ground[i, 0].item()
        estimation = Pred[i, 0].item()

        assert type(second) == int
        assert type(label) == float
        assert type(estimation) == float

        EstimationDict[hashValue].append(estimation / 60.0 * (second / 60.0))
        LabelDict[hashValue].append(label / 60.0 * (second / 60.0))

    for station_time, data in EstimationDict.items():
        RainfallHourlyEstimationList.append(sum(data))
        RainfallHourlyLabelList.append(sum(LabelDict[station_time]))

    assert len(RainfallHourlyEstimationList) == len(RainfallHourlyLabelList)

    print(len(RainfallHourlyEstimationList))
    print(len(RainfallHourlyLabelList))

    with open(os.path.join(saveFolder, "{}_Test_EstimationHourly.pkl".format(modelName)), "wb") as saveHandle_1:
        pickle.dump(RainfallHourlyEstimationList, saveHandle_1)

    with open(os.path.join(saveFolder, "{}_Test_LabelHourly.pkl".format(modelName)), "wb") as saveHandle_2:
        pickle.dump(RainfallHourlyLabelList, saveHandle_2)

    print("Finish Convert.")


if __name__ == '__main__':
    model = "QPET"

    ground = numpy.load(
        "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_GroundTruth_RainRate.npy".format(model, model)
    )

    pred = numpy.load(
        "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_Estimation_RainRate.npy".format(model, model)
    )

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_Filenames.pkl".format(
                model, model
            ), "rb") as handle1:
        filenames = pickle.load(handle1)

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_Scan_Seconds.pkl".format(
                model, model
            ), "rb") as handle2:
        seconds = pickle.load(handle2)

    save = "/media/data3/zhangjianchang/DATA/Estimation/{}".format(model)

    RainRateConvertRainfallHourly(pred, ground, filenames, seconds, save, model)
