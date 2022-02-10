# @Time     : 12/28/2021 5:07 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : RadarGaugeDataset.py
# @Project  : QuantitativePrecipitationEstimation
import os
import pickle
from typing import Callable

import numpy
import pandas
import torch
from torch.utils.data import Dataset


def normalization(fieldData: numpy.ndarray, old_min: float, ole_max: float,
                  new_min: float, new_max: float, nan_value: float = 0.0):

    k = (new_max - new_min) / (ole_max - old_min)
    fieldData = new_min + k * (fieldData-old_min)
    fieldData[numpy.isnan(fieldData)] = nan_value
    fieldData = numpy.clip(fieldData, a_min=new_min, a_max=new_max)
    return fieldData


class DatasetForQuantitativePrecipitationEstimation(Dataset):
    def __init__(self, IndicesPath: str, DataPath: str,
                 isNormalize: bool = True,
                 RainTransform: Callable = None):
        super(DatasetForQuantitativePrecipitationEstimation, self).__init__()
        self.DataPath = DataPath
        self.DataList = pandas.read_csv(IndicesPath, index_col=0)
        self.isNormalize = isNormalize
        self.RainTransform = RainTransform

    def __len__(self):
        return len(self.DataList)

    def __getitem__(self, item):
        Filename = self.DataList.iloc[item]["FILENAME"]
        meanRainRate = self.DataList.iloc[item]["MEAN_OF_RAIN_RATE"].item()
        scanSeconds = self.DataList.iloc[item]["RADAR_SCAN_SECONDS"].item()

        # read pkl file
        with open(os.path.join(self.DataPath, Filename), "rb") as f:
            polarization, polarizationField, rainRateRecord = pickle.load(f)

        # get lowest elevation data
        dBZ_field = polarizationField["reflectivity"][0:2, ...]
        Zdr_field = polarizationField["differential_reflectivity"][0:2, ...]

        # data scale
        if self.isNormalize:
            dBZ_field = normalization(dBZ_field, 0, 65, -0.5, 1.0, nan_value=0.0)
            Zdr_field = normalization(Zdr_field, -0.5, 3.5, -1.0, 1.0, nan_value=0.0)

        else:
            dBZ_field[numpy.isnan(dBZ_field)] = -15.0
            Zdr_field[numpy.isnan(Zdr_field)] = -8.0

        if self.RainTransform is not None:
            meanRainRate = self.RainTransform(meanRainRate)

        # features
        dBZ_Tensor = torch.from_numpy(dBZ_field)
        Zdr_Tensor = torch.from_numpy(Zdr_field)

        field = torch.cat([dBZ_Tensor, Zdr_Tensor], dim=0)
        label = torch.as_tensor([meanRainRate])

        return Filename, scanSeconds, field, label


class ContinuousDatasetForQuantitativePrecipitationEstimation(Dataset):
    def __init__(self, IndicesPath: str, DataPath: str,
                 isNormalize: bool = True,
                 RainTransform: Callable = None):
        super(ContinuousDatasetForQuantitativePrecipitationEstimation, self).__init__()
        self.DataPath = DataPath
        self.DataList = pandas.read_csv(IndicesPath, index_col=0)
        self.isNormalize = isNormalize
        self.RainTransform = RainTransform

    def __len__(self):
        return len(self.DataList)

    def __getitem__(self, item):
        Filename = self.DataList.iloc[item]["FILENAME"]
        meanRainRate = self.DataList.iloc[item]["MEAN_OF_RAIN_RATE"].item()
        scanSeconds = self.DataList.iloc[item]["RADAR_SCAN_SECONDS"].item()
        isNormal = self.DataList.iloc[item]["REASON"]

        if isNormal != "Normal":
            return Filename, 0.0, 0.0, 0.0

        # read pkl file
        with open(os.path.join(self.DataPath, Filename), "rb") as f:
            polarization, polarizationField, rainRateRecord = pickle.load(f)

        # get lowest elevation data
        dBZ_field = polarizationField["reflectivity"][0:2, ...]
        Zdr_field = polarizationField["differential_reflectivity"][0:2, ...]

        # data scale
        if self.isNormalize:
            dBZ_field = normalization(dBZ_field, 0, 65, -0.5, 1.0, nan_value=0.0)
            Zdr_field = normalization(Zdr_field, -0.5, 3.5, -1.0, 1.0, nan_value=0.0)

        else:
            dBZ_field[numpy.isnan(dBZ_field)] = -15.0
            Zdr_field[numpy.isnan(Zdr_field)] = -8.0

        if self.RainTransform is not None:
            meanRainRate = self.RainTransform(meanRainRate)

        # features
        dBZ_Tensor = torch.from_numpy(dBZ_field)
        Zdr_Tensor = torch.from_numpy(Zdr_field)

        field = torch.cat([dBZ_Tensor, Zdr_Tensor], dim=0)
        label = torch.as_tensor([meanRainRate])

        return Filename, scanSeconds, field, label
