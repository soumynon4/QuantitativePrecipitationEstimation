# @Time     : 7/10/2021 9:12 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : Main.py
# @Project  : QuantitativePrecipitationEstimation
import datetime
import os
import pickle
from pprint import pprint

import numpy
import pandas
import pyart

from Gauge import ReadGaugeData
from Radar import radar_time_gauge_time, getRadarData, determine_time_for_radar

locationInformationGPM = {"SFL0010": {"azimuth": 279.38, "range": 58250},
                          "SFL0011": {"azimuth": 191.80, "range": 90190},
                          "SFL0012": {"azimuth": 198.60, "range": 82440},
                          "SFL0024": {"azimuth": 183.07, "range": 94040},
                          "SFL0038": {"azimuth": 275.93, "range": 45480},
                          "SFL0039": {"azimuth": 197.96, "range": 93550},
                          "SFL0041": {"azimuth": 170.57, "range": 84220},
                          "SFL0052": {"azimuth": 218.08, "range": 86380},
                          "SFL0065": {"azimuth": 190.68, "range": 86090},
                          "SFL0067": {"azimuth": 187.81, "range": 96570},
                          "SFL0068": {"azimuth": 187.72, "range": 88900},
                          "SFL0085": {"azimuth": 272.48, "range": 68510},
                          "SFL0097": {"azimuth": 290.61, "range": 83460},
                          "SFL0101": {"azimuth": 156.69, "range": 99760},
                          "SFL0109": {"azimuth": 264.35, "range": 73550},
                          "SFL0115": {"azimuth": 167.80, "range": 89150},
                          "SFL0116": {"azimuth": 178.36, "range": 84140},
                          "SFL0145": {"azimuth": 162.93, "range": 99170},
                          "SFL0157": {"azimuth": 165.90, "range": 73770},
                          "SFL0158": {"azimuth": 275.93, "range": 64940},
                          "SFL0167": {"azimuth": 273.51, "range": 75740},
                          "SFL0171": {"azimuth": 202.18, "range": 96070},
                          "SFL0172": {"azimuth": 227.73, "range": 70750},
                          "SFL0204": {"azimuth": 228.26, "range": 80530},
                          "SFL0205": {"azimuth": 204.01, "range": 86490},
                          "SFL0214": {"azimuth": 306.34, "range": 63760},
                          "SFL0218": {"azimuth": 264.02, "range": 79830},
                          "SFL0221": {"azimuth": 226.04, "range": 57810},
                          "SFL0228": {"azimuth": 201.49, "range": 74000},
                          "SFL0230": {"azimuth": 241.28, "range": 75410},
                          "SFL0231": {"azimuth": 280.78, "range": 84860},
                          "SFL0232": {"azimuth": 235.07, "range": 43630},
                          "SFL0235": {"azimuth": 232.92, "range": 96370},
                          "SFL0237": {"azimuth": 217.23, "range": 60080},
                          "SFL0238": {"azimuth": 214.60, "range": 77350},
                          "SFL0239": {"azimuth": 215.91, "range": 92730},
                          "SFL0244": {"azimuth": 210.07, "range": 72530},
                          "SFL0248": {"azimuth": 195.42, "range": 64140},
                          "SFL0284": {"azimuth": 237.15, "range": 63680},
                          "SFL0295": {"azimuth": 258.13, "range": 76640},
                          "SFL0308": {"azimuth": 222.99, "range": 69080},
                          "SFL0309": {"azimuth": 164.20, "range": 82880}}

# _RAIN_DATA_PATH = "c:\\users\\admin\\desktop\\rain"
# _CSV_FOLDER_PATH = "c:\\users\\admin\\desktop\\csv"

_RAIN_DATA_PATH = "/media/data3/zhangjianchang/DATA/Dataset/RAIN"
_CSV_FOLDER_PATH = "/media/data3/zhangjianchang/DATA/Dataset/CSV"


def CreateData(targetYear: int, radarDirectory: str, gaugeDirectory: str):
    radarFileList = sorted(os.listdir(radarDirectory))

    _FILE_NAME_LIST = list()
    _RAIN_LIST = list()
    _RAIN_RATE_LIST = list()
    _RADAR_START = list()
    _RADAR_END = list()
    _SCAN_SECONDS = list()

    for singleRadarFile in radarFileList:
        singleRadarPath = os.path.join(radarDirectory, singleRadarFile)

        # 处理雷达数据
        # 1. 如果雷达数据无法正常打开, 跳过该雷达数据
        # 2. 如果雷达扫描持续时间过短, 跳过该数据

        try:
            radar = pyart.io.read_nexrad_archive(singleRadarPath)
        except Exception:
            print("{} ending process, Reason: Radar file can't open".format(singleRadarFile))
            continue

        if radar.fixed_angle["data"].size <= 4:
            print("{} ending process, Reason: Radar Scan Sweep <= 4".format(singleRadarFile))
            continue

        if radar.time["data"][-1].item() <= 60:
            print("{} ending process, Reason: Radar Scan Time <= 60s".format(singleRadarFile))
            continue

        gaugeStartTime, gaugeEndTime = radar_time_gauge_time(singleRadarFile, radar.time["data"][-1].item())
        radarStartTime = determine_time_for_radar(singleRadarFile)
        radarEndTime = radarStartTime + datetime.timedelta(seconds=int(radar.time["data"][-1].item()))

        # 根据雷达扫描的开始和结束时间, 确定时间范围内的雨量站数据时间
        print("Gauge data should start: {}, should end: {}".format(gaugeStartTime, gaugeEndTime))

        # 根据雷达来匹配雨量站数据
        for gaugeName, azimuthAndRange in locationInformationGPM.items():
            print("{} {}::{}::{} Process Start {}".format(
                '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
            ))
            rainRateRecord = ReadGaugeData(gaugeStartTime, gaugeEndTime, gaugeName, gaugeDirectory)

            if sum(rainRateRecord) / len(rainRateRecord) <= 1.0:
                print("Reason: mean rain rate <= 1.0 mm/h")
                print("{} {}::{}::{} Process End {}\n".format(
                    '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
                ))
                continue

            if not rainRateRecord:
                print("Reason: There is no rain rate data")
                print("{} {}::{}::{} Process End {}\n".format(
                    '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
                ))
                continue

            elif len(rainRateRecord) < (gaugeEndTime - gaugeStartTime).seconds // 60:
                print("Reason: {} records less than {}".format(
                    len(rainRateRecord), (gaugeEndTime - gaugeStartTime).seconds // 60
                ))

                print("{} {}::{}::{} Process End {}\n".format(
                    '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
                ))
                continue

            polarization, polarization_field = getRadarData(
                radar,
                start_time=determine_time_for_radar(singleRadarFile),
                target_azimuth=azimuthAndRange["azimuth"],
                target_range=float(azimuthAndRange["range"])
            )

            if len(polarization) == 0 or len(polarization_field) == 0:
                print("Reason: Radar data is less than the needed number")
                print("{} {}::{}::{} Process End {}\n".format(
                    '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
                ))
                continue

            if numpy.all(numpy.isnan(polarization["reflectivity"])):
                print("Reason: Radar data is all Nan")
                print('{} Rain Events in Time Scale:'.format(gaugeName))
                print('{} --- {}'.format(gaugeStartTime, gaugeEndTime))
                pprint(rainRateRecord)
                print("{} {}::{}::{} Process End {}\n".format(
                    '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
                ))
                continue

            print('{} Rain Events in Time Scale:'.format(gaugeName))
            print('{} --- {}'.format(gaugeStartTime, gaugeEndTime))
            pprint(rainRateRecord)
            # print('Calculated Rain Rate : {} [mm/h]'.format())

            with open(os.path.join(_RAIN_DATA_PATH, "{}_{}.pkl".format(singleRadarFile[4:-4], gaugeName)), "wb") as f:
                pickle.dump((polarization, polarization_field, rainRateRecord), f)

            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeName))
            _RAIN_LIST.append(sum(rainRateRecord))
            _RAIN_RATE_LIST.append(sum(rainRateRecord) / len(rainRateRecord))
            _RADAR_START.append(radarStartTime)
            _RADAR_END.append(radarEndTime)
            _SCAN_SECONDS.append((radarEndTime - radarStartTime).seconds)

            print("Reason: Complete")
            print("{} {}::{}::{} Process End {}\n".format(
                '>' * 10, targetYear, gaugeName, singleRadarFile, '<' * 10
            ))

    assert len(_FILE_NAME_LIST) == len(_RAIN_LIST)

    _INDEX_LIST = list(range(len(_FILE_NAME_LIST)))

    df = pandas.DataFrame({'FILENAME': pandas.Series(data=_FILE_NAME_LIST, index=_INDEX_LIST),
                           'SUM_OF_RAIN_RATE': pandas.Series(data=_RAIN_LIST, index=_INDEX_LIST, dtype='float'),
                           'MEAN_OF_RAIN_RATE': pandas.Series(data=_RAIN_RATE_LIST, index=_INDEX_LIST, dtype='float'),
                           'RADAR_SCAN_START': pandas.Series(data=_RADAR_START, index=_INDEX_LIST),
                           'RADAR_SCAN_END': pandas.Series(data=_RADAR_END, index=_INDEX_LIST),
                           'RADAR_SCAN_SECONDS': pandas.Series(data=_SCAN_SECONDS, index=_INDEX_LIST, dtype='int')
                           })
    # SUM_OF_RAIN_RATE
    # 雷达扫描时间内的某个雨量站数据相加之和
    # 除以60为雷达扫面时间的降水量

    # MEAN_OF_RAIN_RATE
    # rain rate record 的均值

    df.to_csv(os.path.join(_CSV_FOLDER_PATH, '{}.csv'.format(targetYear)))


if __name__ == '__main__':
    CreateData(2020,
               "/media/data3/zhangjianchang/RadarData/2020",
               "/media/data3/zhangjianchang/DATA/Gauge/ProcessedGaugeCSV")