# @Time     : 12/27/2021 2:10 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : Metric.py
# @Project  : QuantitativePrecipitationEstimation
import numpy as np


def BIAS(estimations: np.ndarray, labels: np.ndarray):
    estimations = np.squeeze(estimations)
    labels = np.squeeze(labels)
    if estimations.ndim != 1 or labels.ndim != 1:
        raise ValueError("dimension error!")
    assert estimations.shape[0] == labels.shape[0]

    bias = np.sum(estimations) / np.sum(labels)
    return bias.item()


def NSE(estimations: np.ndarray, labels: np.ndarray):
    estimations = np.squeeze(estimations)
    labels = np.squeeze(labels)
    if estimations.ndim != 1 or labels.ndim != 1:
        raise ValueError("dimension error!")
    assert estimations.shape[0] == labels.shape[0]

    nse = np.sum(np.abs(estimations - labels)) / np.sum(labels)
    return nse.item()


def MAE(estimations, labels):
    estimations = np.array(estimations)
    labels = np.array(labels)

    estimations = np.squeeze(estimations)
    labels = np.squeeze(labels)
    if estimations.ndim != 1 or labels.ndim != 1:
        raise ValueError("dimension error!")
    assert estimations.shape[0] == labels.shape[0]

    mae = np.sum(np.abs(estimations - labels)) / estimations.shape[0]
    return mae.item()


def RMSE(estimations: np.ndarray, labels: np.ndarray):
    estimations = np.squeeze(estimations)
    labels = np.squeeze(labels)
    if estimations.ndim != 1 or labels.ndim != 1:
        raise ValueError("dimension error!")
    assert estimations.shape[0] == labels.shape[0]

    rmse = np.sqrt(np.sum(np.square(estimations - labels)) / estimations.shape[0])
    return rmse.item()


metrics = {
    'BIAS': BIAS,
    'NSE': NSE,
    "RMSE": RMSE,
    "MAE": MAE,
    # could add more metrics such as accuracy for each token type
}
