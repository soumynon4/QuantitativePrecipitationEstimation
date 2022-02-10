# @Time     : 12/28/2021 3:52 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : SplitData.py
# @Project  : QuantitativePrecipitationEstimation
import os
import math
from typing import Iterable

import numpy as np
import pandas


def generateRandomIndex(totalSize: int, validateRatio: float, testRatio: float):
    np.random.seed(3723011)
    testNumber = math.floor(totalSize * testRatio)
    validateNumber = math.floor(totalSize * validateRatio)

    randomArray = np.arange(totalSize)
    np.random.shuffle(randomArray)
    testIndices = randomArray[:testNumber]
    validateIndices = randomArray[testNumber:testNumber + validateNumber]
    trainIndices = randomArray[testNumber + validateNumber:]

    return trainIndices, validateIndices, testIndices


def createDataset(csvFilePaths: Iterable[str],
                  saveFolder: str,
                  validateFileRatio: float = 0.1,
                  testFileRatio: float = 0.2):

    np.random.seed(0)
    dataFrameList = list()
    totalNumber = 0
    for path in csvFilePaths:
        singleDataFrame = pandas.read_csv(path, index_col=0)
        dataFrameList.append(singleDataFrame)
        print("File: {} Number: {}".format(path, len(singleDataFrame)))
        totalNumber += len(singleDataFrame)

    print("Total Number: {}".format(totalNumber))

    dataFrameAll = pandas.concat(dataFrameList, axis=0, ignore_index=True)
    print(dataFrameAll)

    trainDataIndex, validateDataIndex, testDataIndex = generateRandomIndex(totalNumber, validateFileRatio,
                                                                           testFileRatio)

    trainDataset = dataFrameAll.iloc[trainDataIndex].reset_index(drop=True)
    validateDataset = dataFrameAll.iloc[validateDataIndex].reset_index(drop=True)
    testDataset = dataFrameAll.iloc[testDataIndex].reset_index(drop=True)

    print("Train Dataset Number: {}".format(len(trainDataset)))
    print("Validate Dataset Number: {}".format(len(validateDataset)))
    print("Test Dataset Number: {}".format(len(testDataset)))

    trainDataset.to_csv(os.path.join(saveFolder, "TrainDataset.csv"))
    validateDataset.to_csv(os.path.join(saveFolder, "ValidateDataset.csv"))
    testDataset.to_csv(os.path.join(saveFolder, "TestDataset.csv"))


if __name__ == '__main__':
    createDataset(
        ["/media/data3/zhangjianchang/DATA/Dataset/CSV/2017.csv",
         "/media/data3/zhangjianchang/DATA/Dataset/CSV/2018.csv",
         "/media/data3/zhangjianchang/DATA/Dataset/CSV/2019.csv",
         "/media/data3/zhangjianchang/DATA/Dataset/CSV/2020.csv"],
        "/media/data3/zhangjianchang/DATA/Dataset/CSV/"
    )
