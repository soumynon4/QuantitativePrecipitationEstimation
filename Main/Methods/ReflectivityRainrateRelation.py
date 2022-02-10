# @Time     : 12/22/2021 9:25 AM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ReflectivityRainrateRelation.py
# @Project  : QuantitativePrecipitationEstimation
import numpy


def dBZToReflectivityFactor(dBZ: numpy.ndarray):
    Z = 10 ** (dBZ / 10)
    return Z


def QPEbyZh(dBZ: numpy.ndarray, coefficient: int = 300, index: float = 1.4):
    # [200, 1.6]
    # [300, 1.4]
    # [250, 1.2]
    Z = dBZToReflectivityFactor(dBZ)
    estimation = (Z / coefficient) ** (1 / index)
    return estimation


def QPEbyZhAndZdr(dBZ: numpy.ndarray, Zdr: numpy.ndarray):
    assert dBZ.shape == Zdr.shape
    Z = dBZToReflectivityFactor(dBZ)
    estimation = 0.0142 * (Z ** 0.77) * ((10 ** (Zdr/10)) ** -1.67)
    return estimation


def ma_dBZToReflectivityFactor(dBZ: numpy.ma.MaskedArray):
    Z = 10 ** (dBZ / 10)
    return Z


def ma_QPEbyZh(dBZ: numpy.ma.MaskedArray, coefficient: float = 300.0, index: float = 1.4):
    # [200, 1.6]
    # [300, 1.4]
    # [250, 1.2]
    Z = ma_dBZToReflectivityFactor(dBZ)
    estimation = (Z / coefficient) ** (1 / index)
    return estimation


def ma_QPEbyZhAndZdr(dBZ: numpy.ma.MaskedArray, Zdr: numpy.ma.MaskedArray):
    assert dBZ.shape == Zdr.shape
    Z = ma_dBZToReflectivityFactor(dBZ)
    estimation = 0.0142 * (Z ** 0.77) * ((10 ** (Zdr / 10)) ** -1.67)
    return estimation
