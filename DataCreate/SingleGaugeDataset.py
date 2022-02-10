# @Time     : 2/7/2022 12:18 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : SingleGaugeDataset.py
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
from Main import locationInformationGPM


_RAIN_DATA_PATH = "c:\\users\\admin\\downloads\\rain"
_CSV_FOLDER_PATH = "c:\\users\\admin\\downloads\\csv"


def SingleGaugeDatasetCreate(targetTime: str, radarDirectory: str, gaugeDirectory: str, gaugeStationName: str):
    radarFileList = sorted(os.listdir(radarDirectory))

    _FILE_NAME_LIST = list()
    _RAIN_LIST = list()
    _RAIN_RATE_LIST = list()
    _RADAR_START = list()
    _RADAR_END = list()
    _SCAN_SECONDS = list()
    _REASON = list()

    for singleRadarFile in radarFileList:
        singleRadarPath = os.path.join(radarDirectory, singleRadarFile)

        # 处理雷达数据
        # 1. 如果雷达数据无法正常打开, 跳过该雷达数据
        # 2. 如果雷达扫描持续时间过短, 跳过该数据

        try:
            radar = pyart.io.read_nexrad_archive(singleRadarPath)
        except Exception:
            print("{} ending process, Reason: Radar file can't open".format(singleRadarFile))
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(0.0)
            _RADAR_END.append(0.0)
            _SCAN_SECONDS.append(0.0)
            _REASON.append("can't open")
            continue

        if radar.fixed_angle["data"].size <= 4:
            print("{} ending process, Reason: Radar Scan Sweep <= 4".format(singleRadarFile))
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(0.0)
            _RADAR_END.append(0.0)
            _SCAN_SECONDS.append(0.0)
            _REASON.append("radar scan sweep less than 4")
            continue

        if radar.time["data"][-1].item() <= 60:
            print("{} ending process, Reason: Radar Scan Time <= 60s".format(singleRadarFile))
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(0.0)
            _RADAR_END.append(0.0)
            _SCAN_SECONDS.append(0.0)
            _REASON.append("radar scan time less than 60s")
            continue

        gaugeStartTime, gaugeEndTime = radar_time_gauge_time(singleRadarFile, radar.time["data"][-1].item())
        radarStartTime = determine_time_for_radar(singleRadarFile)
        radarEndTime = radarStartTime + datetime.timedelta(seconds=int(radar.time["data"][-1].item()))

        # 根据雷达扫描的开始和结束时间, 确定时间范围内的雨量站数据时间
        print("Gauge data should start: {}, should end: {}".format(gaugeStartTime, gaugeEndTime))

        # 根据雷达来匹配雨量站数据
        azimuthAndRange = locationInformationGPM[gaugeStationName]
        print("{} {} Process Start {}".format(targetTime, gaugeStationName, singleRadarFile))

        rainRateRecord = ReadGaugeData(gaugeStartTime, gaugeEndTime, gaugeStationName, gaugeDirectory)

        if not rainRateRecord:
            print("Reason: There is no rain rate data")
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(radarStartTime)
            _RADAR_END.append(radarEndTime)
            _SCAN_SECONDS.append((radarEndTime - radarStartTime).seconds)
            _REASON.append("gauge data empty")
            continue

        elif len(rainRateRecord) < (gaugeEndTime - gaugeStartTime).seconds // 60:
            print("Reason: {} records less than {}".format(
                len(rainRateRecord), (gaugeEndTime - gaugeStartTime).seconds // 60
            ))
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(radarStartTime)
            _RADAR_END.append(radarEndTime)
            _SCAN_SECONDS.append((radarEndTime - radarStartTime).seconds)
            _REASON.append("{} records less than {}".format(
                len(rainRateRecord), (gaugeEndTime - gaugeStartTime).seconds // 60
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
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(radarStartTime)
            _RADAR_END.append(radarEndTime)
            _SCAN_SECONDS.append((radarEndTime - radarStartTime).seconds)
            _REASON.append("radar data is less than the needed number")
            continue

        if numpy.all(numpy.isnan(polarization["reflectivity"])):
            print("Reason: Radar data is all Nan")
            _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
            _RAIN_LIST.append(0.0)
            _RAIN_RATE_LIST.append(0.0)
            _RADAR_START.append(radarStartTime)
            _RADAR_END.append(radarEndTime)
            _SCAN_SECONDS.append((radarEndTime - radarStartTime).seconds)
            _REASON.append("radar data is all Nan")
            continue

        print('{} Rain Events in Time Scale:'.format(gaugeStationName))
        print('{} --- {}'.format(gaugeStartTime, gaugeEndTime))
        pprint(rainRateRecord)
        # print('Calculated Rain Rate : {} [mm/h]'.format())

        with open(os.path.join(_RAIN_DATA_PATH, "{}_{}.pkl".format(singleRadarFile[4:-4], gaugeStationName)), "wb") as f:
            pickle.dump((polarization, polarization_field, rainRateRecord), f)

        _FILE_NAME_LIST.append('{}_{}.pkl'.format(singleRadarFile[4:-4], gaugeStationName))
        _RAIN_LIST.append(sum(rainRateRecord))
        _RAIN_RATE_LIST.append(sum(rainRateRecord) / len(rainRateRecord))
        _RADAR_START.append(radarStartTime)
        _RADAR_END.append(radarEndTime)
        _SCAN_SECONDS.append((radarEndTime - radarStartTime).seconds)
        _REASON.append("Normal")

    print("Reason: Complete")
    assert len(_FILE_NAME_LIST) == len(radarFileList)
    _INDEX_LIST = list(range(len(_FILE_NAME_LIST)))
    df = pandas.DataFrame({'FILENAME': pandas.Series(data=_FILE_NAME_LIST, index=_INDEX_LIST),
                           'SUM_OF_RAIN_RATE': pandas.Series(data=_RAIN_LIST, index=_INDEX_LIST, dtype='float'),
                           'MEAN_OF_RAIN_RATE': pandas.Series(data=_RAIN_RATE_LIST, index=_INDEX_LIST, dtype='float'),
                           'RADAR_SCAN_START': pandas.Series(data=_RADAR_START, index=_INDEX_LIST),
                           'RADAR_SCAN_END': pandas.Series(data=_RADAR_END, index=_INDEX_LIST),
                           'RADAR_SCAN_SECONDS': pandas.Series(data=_SCAN_SECONDS, index=_INDEX_LIST, dtype='int'),
                           'REASON': pandas.Series(data=_REASON, index=_INDEX_LIST)
                           })

    # SUM_OF_RAIN_RATE
    # 雷达扫描时间内的某个雨量站数据相加之和
    # 除以60为雷达扫面时间的降水量
    # MEAN_OF_RAIN_RATE
    # rain rate record 的均值
    df.to_csv(os.path.join(_CSV_FOLDER_PATH, '{}_{}.csv'.format(gaugeStationName, targetTime)))


if __name__ == '__main__':
    SingleGaugeDatasetCreate("20210705",
                             "c:\\users\\admin\\downloads\\radar",
                             "c:\\users\\admin\\downloads\\ProcessedGaugeCSV",
                             "SFL0309")
