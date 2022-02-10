# @Time     : 12/27/2021 2:01 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ProcessOfData.py
# @Project  : QuantitativePrecipitationEstimation
import math

import numpy as np


def dBZToInput(dBZ: np.ndarray, setMin: float = -15.0, setMax: float = 60.0, reverse: bool = False):
    if not reverse:
        dBZ[np.isnan(dBZ)] = setMin
        return np.clip((dBZ - setMin) / (setMax - setMin), a_min=0.0, a_max=1.0)
    else:
        return dBZ * (setMax - setMin) + setMin


def ZdrToInput(Zdr: np.ndarray, setMin: float = -8.0, setMax: float = 5.0, reverse: bool = False):
    if not reverse:
        Zdr[np.isnan(Zdr)] = setMin
        Zdr[Zdr > setMax] = setMin
        return np.clip((Zdr - setMin) / (setMax - setMin), a_min=0.0, a_max=1.0)
    else:
        return Zdr * (setMax - setMin) + setMin


def SUM_OF_RAIN_RATEToLabel(sumOfRainRate: float, seconds: int, reverse: bool = False):
    if not reverse:
        outcome = sumOfRainRate / 60 / seconds * 1000
        outcome = math.log2(1 + outcome) * 10
        return outcome
    else:
        return (2 ** (sumOfRainRate / 10) - 1) * 60 * seconds / 1000


def MEAN_OF_RAIN_RATEToLabel(meanOfRainRate: float, reverse: bool = False):
    if not reverse:
        return math.log10(1 + meanOfRainRate) * 10
    else:
        return 10 ** (meanOfRainRate / 10) - 1
