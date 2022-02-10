# @Time     : 2/8/2022 3:46 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ContinuousPrecipitationPlot.py
# @Project  : QuantitativePrecipitationEstimation
import sys
import os
import time as t

import numpy as np
import matplotlib.pyplot as plt
import pyart

sys.path.append("/media/data3/zhangjianchang/QuantitativePrecipitationEstimation")
from DataCreate.Gauge import ReadGaugeData
from DataCreate.Radar import radar_time_gauge_time
from DataCreate.Radar import determine_time_for_radar


def singleGaugeStationRainFallHourlyPlot(subplot, coordinate, rainFallData, color, lineStyle, linewidth=2):
    subplot.plot(coordinate, rainFallData, color=color, linestyle=lineStyle, linewidth=linewidth)


def PointWiseRainFall(radarDirectory: str,
                      estimationDirectory: str,
                      processedGaugeDirectoryCSV: str,
                      gaugeList: list,
                      calculateMethod: str,
                      pictureSavePath: str = "/media/data3/zhangjianchang/DATA/Pictures/KMLB/PointWise"):
    LabelFontDict = {"family": "Arial",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 15}
    TitleFontDict = {"family": "Arial",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 10}
    radarFiles = sorted(os.listdir(radarDirectory))

    for gaugeStationName in gaugeList:

        MainFigure = plt.figure(dpi=150)
        SubFigure = mainFigure.add_subplot(1, 1, 1)

        thisGaugeStationEstimations = np.load(os.path.join(estimationDirectory,
                                                           "{}-{}-Estimation.npy".
                                                           format(gaugeStationName, calculateMethod)))
        radarFilesIndex = 0
        EstimationHourly = []
        GaugeHourly = []

        totalSeconds = 0
        EstimationOneHour = 0
        GaugeOneHour = 0
        for radarFile in radarFiles:
            thisRadar = pyart.io.read_nexrad_archive(os.path.join(radarDirectory, radarFile))
            thisScanStartTime, thisScanEndTime = radar_time_gauge_time(radarFile, thisRadar.time["data"][-1].item())
            rainRateRecords = ReadGaugeData(thisScanStartTime, thisScanEndTime, gaugeStationName,
                                            processedGaugeDirectoryCSV)
            GaugeOneHour += (sum(rainRateRecords) / 60)
            EstimationOneHour += (thisGaugeStationEstimations[radarFilesIndex].item() * (
                    thisScanEndTime - thisScanStartTime).seconds / 3600)
            totalSeconds += (thisScanEndTime - thisScanStartTime).seconds
            if totalSeconds >= 3600:
                EstimationHourly.append(EstimationOneHour)
                GaugeHourly.append(GaugeOneHour)

                totalSeconds = 0
                EstimationOneHour = 0
                GaugeOneHour = 0

            radarFilesIndex += 1

        print(GaugeHourly)
        print(EstimationHourly)
        t.sleep(100)

        startGaugeIndex = 0
        endGaugeIndex = len(GaugeHourly) - 1
        for i in range(len(GaugeHourly)):
            if GaugeHourly[i] == 0.0:
                continue
            else:
                startGaugeIndex = i
                break

        for i in range(len(GaugeHourly)):
            if GaugeHourly[endGaugeIndex - i] == 0.0:
                continue
            else:
                endGaugeIndex -= i
                break
        GaugeHourly = GaugeHourly[startGaugeIndex: endGaugeIndex + 1]
        EstimationHourly = EstimationHourly[startGaugeIndex: endGaugeIndex + 1]
        titleStartTime = determine_time_for_radar(radarFiles[startGaugeIndex])
        titleEndTime = determine_time_for_radar(radarFiles[endGaugeIndex])

        plt.title(
            'PrecipitationHourly ({}) \n{:4}:{:02}:{:02}:{:02}:{:02}:{:02}UTC - {:4}:{:02}:{:02}:{:02}:{:02}:{:02}UTC'.
                format(gaugeStationName,
                       titleStartTime.year, titleStartTime.month, titleStartTime.day,
                       titleStartTime.hour, titleStartTime.minute, titleStartTime.second,
                       titleEndTime.year, titleEndTime.month, titleEndTime.day,
                       titleEndTime.hour, titleEndTime.minute, titleEndTime.second),
            fontdict=titleFontDict)

        subFigure.set_xlabel("Time [hour]", fontdict=labelFontDict)
        subFigure.set_ylabel("Rain Fall [mm]", fontdict=labelFontDict)
        subFigure.set_yticks(np.arange(0, 35, 5))
        subFigure.set_xticks(np.arange(1, len(GaugeHourly) + 1, 1))
        plt.xlim(1, len(GaugeHourly))
        plt.ylim(0, 30)
        time = np.arange(1, len(GaugeHourly) + 1, 1)
        singleGaugeStationRainFallHourlyPlot(subFigure, time, GaugeHourly, "red", '-', "Gauge")
        singleGaugeStationRainFallHourlyPlot(subFigure, time, EstimationHourly, "black", '-', "DeepLearningEstimations")
        plt.legend(loc='best')
        plt.savefig(os.path.join(pictureSavePath, "{}_PointWiseContinuous_{:4}:{:02}:{:02}.png".
                                 format(gaugeStationName,
                                        titleStartTime.year, titleStartTime.month, titleStartTime.day)))
        plt.clf()
        break


if __name__ == '__main__':
    # radarDirectory = "/media/data3/zhangjianchang/DATA/RadarData/KMLB/20170911"
    # estimationDirectory = "/media/data3/zhangjianchang/DATA/Estimation/KMLB/PointWise/20170911"
    # processedGaugeDirectoryCSV = "/media/data3/zhangjianchang/DATA/GPM/ProcessedGaugeCSV"
    # gaugeList = ['SFL0011', 'SFL0024', 'SFL0039', 'SFL0065', 'SFL0067',
    #              'SFL0068', 'SFL0101', 'SFL0115', 'SFL0116', 'SFL0157',
    #              'SFL0171', 'SFL0228', 'SFL0237', 'SFL0238', 'SFL0244',
    #              'SFL0248', 'SFL0308', 'SFL0309']

    # PointWiseRainFall("/media/data3/zhangjianchang/DATA/RadarData/KMLB/20170911",
    #                   "/media/data3/zhangjianchang/DATA/PrecipitationEvents/KMLB/PointWise/20170911",
    #                   "/media/data3/zhangjianchang/DATA/GPM/ProcessedGaugeCSV",
    #                   ['SFL0011'],
    #                   "FullyConnect")

    labelFontDict = {"family": "Arial",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 20}
    titleFontDict = {"family": "Arial",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 25}
    textFontDict = {"family": "Arial",
                    "style": "normal",
                    "weight": "bold",
                    "color": "black",
                    "size": 15}

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_LabelHourly.pkl".format(
                model, model
            ), "rb") as handle2:
        label = np.array(pickle.load(handle2))

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_EstimationHourly.pkl".format(
                model, model
            ), "rb") as handle1:
        qpetEstimation = np.array(pickle.load(handle1))

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_EstimationHourly.pkl".format(
                model, model
            ), "rb") as handle1:
        mlpEstimation = np.array(pickle.load(handle1))

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_LabelHourly.pkl".format(
                model, model
            ), "rb") as handle2:
        z200Estimation = np.array(pickle.load(handle2))

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_LabelHourly.pkl".format(
                model, model
            ), "rb") as handle2:
        z300Estimation = np.array(pickle.load(handle2))

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_EstimationHourly.pkl".format(
                model, model
            ), "rb") as handle1:
        zh_zdrEstimation = np.array(pickle.load(handle1))




    mainFigure = plt.figure(figsize=(18, 15), dpi=150)
    subFigure = mainFigure.add_subplot(1, 1, 1)
    subFigure.set_xlabel("Time [hour]", fontdict=labelFontDict)
    subFigure.set_ylabel("Rain Fall [mm]", fontdict=labelFontDict)
    subFigure.set_yticks(np.arange(0, 55, 5))
    subFigure.set_xticks(np.arange(1, 19, 1))
    plt.xlim(1, 18)
    plt.ylim(0, 50)
    time = np.arange(1, 19, 1)
    singleGaugeStationRainFallHourlyPlot(subFigure, time, label, "red", '-')
    singleGaugeStationRainFallHourlyPlot(subFigure, time, estimation, "black", '-')
    singleGaugeStationRainFallHourlyPlot(subFigure, time, z200Estimation, "orangered", '--')
    singleGaugeStationRainFallHourlyPlot(subFigure, time, z300Estimation, "darkgreen", '--')
    singleGaugeStationRainFallHourlyPlot(subFigure, time, z250Estimation, "olivedrab", '--')
    singleGaugeStationRainFallHourlyPlot(subFigure, time, zh_zdrEstimation, "blue", '--')

    plt.text(1, 50 - 2, "Deep learning method Corr. = {:.2f},".format(0.9254681411238959, ), fontdict=textFontDict)
    plt.text(1, 50 - 4, "Z-R relation:(200, 1.6) Corr. = {:.2f},".format(0.8678368117448844), fontdict=textFontDict)
    plt.text(1, 50 - 6, "Z-R relation:(300, 1.4) Corr. = {:.2f},".format(0.8587970506724962), fontdict=textFontDict)
    plt.text(1, 50 - 8, "Z-R relation:(250, 1.2) Corr. = {:.2f},".format(0.8434231282715071), fontdict=textFontDict)
    plt.text(1, 50 - 10, "Zh and Zdr method Corr. = {:.2f},".format(0.8712792031127403, ), fontdict=textFontDict)

    plt.text(11, 50 - 2, "MAE = {:.2f}".format(2.9100180705564993), fontdict=textFontDict)
    plt.text(11, 50 - 4, "MAE = {:.2f}".format(5.061311710630815), fontdict=textFontDict)
    plt.text(11, 50 - 6, "MAE = {:.2f}".format(4.924420560394872), fontdict=textFontDict)
    plt.text(11, 50 - 8, "MAE = {:.2f}".format(3.160051548776726), fontdict=textFontDict)
    plt.text(11, 50 - 10, "MAE = {:.2f}".format(4.715105073542148), fontdict=textFontDict)

    plt.title('PrecipitationHourly (SFL0011) \n2017-09-10 01:10:09UTC - 2017-09-11 06:32:48UTC', fontdict=titleFontDict)
    plt.legend(["Gauge", "DeepLearningEstimations", "Z-R relation:(200, 1.6)", "Z-R relation:(300, 1.4)",
                "Z-R relation:(250, 1.2)", "Zh and Zdr"], fontsize=20)
    plt.show()
