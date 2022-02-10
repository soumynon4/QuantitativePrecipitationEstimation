# @Time     : 1/7/2022 6:55 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ConfusionMatrix.py
# @Project  : QuantitativePrecipitationEstimation
import os
import pickle

import numpy


def CalculateConfusionMatrix(pred: numpy.ndarray, label: numpy.ndarray, threshold: float):
    """The following measurements will be used to measure the score of the forecaster
       See Also
       [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
       http://www.wxonline.info/topics/verif2.html

       We will denote
       (a b    (hits       false alarms
        c d) =  misses   correct negatives)

       We will report the
       POD = a / (a + c)
       FAR = b / (a + b)
       CSI = a / (a + b + c)
       Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
       Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
           HSS = 2 * GSS / (GSS + 1)
       """

    TruePositive_hit = numpy.logical_and(pred >= threshold, label >= threshold).sum()
    TrueNegative_correct_negatives = numpy.logical_and(pred < threshold, label < threshold).sum()
    FalsePositive_false_alarm = numpy.logical_and(pred >= threshold, label < threshold).sum()
    FalseNegative_miss = numpy.logical_and(pred < threshold, label >= threshold).sum()

    POD = TruePositive_hit / (TruePositive_hit + FalseNegative_miss)
    FAR = FalsePositive_false_alarm / (TruePositive_hit + FalsePositive_false_alarm)
    CSI = TruePositive_hit / (TruePositive_hit + FalsePositive_false_alarm + FalseNegative_miss)
    HSS = 2 * (TruePositive_hit * TrueNegative_correct_negatives - FalsePositive_false_alarm * FalseNegative_miss) / (
        (TruePositive_hit + FalseNegative_miss) * (FalseNegative_miss + TrueNegative_correct_negatives) +
        (TruePositive_hit + FalsePositive_false_alarm) * (FalsePositive_false_alarm + TrueNegative_correct_negatives)
    )

    GSS = HSS / (2 - HSS)
    return POD, FAR, CSI, HSS, GSS


if __name__ == '__main__':
    modelName = "ZR200"
    threshold = 1.25
    with open("/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_EstimationHourly.pkl".format(
            modelName, modelName), "rb") as handle1:
        prediction = numpy.array(pickle.load(handle1))

    with open("/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_LabelHourly.pkl".format(
            modelName, modelName), "rb") as handle2:
        groundTruth = numpy.array(pickle.load(handle2))

    pod, far, csi, hss, gss = CalculateConfusionMatrix(prediction, groundTruth, threshold=threshold)

    with open(os.path.join("/media/data3/zhangjianchang/DATA/Index/{}".format(modelName), "{}_{}_Index.txt".format(
            modelName, threshold)), "w") as f:
        f.write("threshold: {}\n".format(threshold))
        f.write("POD: {}\n".format(pod))
        f.write("FAR: {}\n".format(far))
        f.write("CSI: {}\n".format(csi))
        f.write("HSS: {}\n".format(hss))
        f.write("GSS: {}\n".format(gss))
