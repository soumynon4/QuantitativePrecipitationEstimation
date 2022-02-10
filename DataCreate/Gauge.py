# @Time     : 7/10/2021 9:11 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : Gauge.py
# @Project  : QuantitativePrecipitationEstimation
import datetime
import os

import numpy
import pandas


def ReadGaugeData(start: datetime.datetime, end: datetime.datetime, station: str, directoryProcessedGaugeData: str) -> list:
    if start.year != end.year:
        raise AttributeError
    else:
        rain_rate_data_list = []
        year = str(start.year)
        try:
            dataFrame = pandas.read_csv(os.path.join(directoryProcessedGaugeData, "{}_{}.csv".format(station, year[-2:])), index_col=0)
        except FileNotFoundError:
            print("Gauge CSV File not found.")
            return rain_rate_data_list

        timeDelta = int((end - start).seconds / 60) + 1
        for i in range(timeDelta):
            data_time = start + datetime.timedelta(minutes=1) * i
            try:
                data = dataFrame.loc[str(data_time), "RainRate"].item()
            except KeyError:
                continue
            rain_rate_data_list.append(data)
        return rain_rate_data_list


def MakeGaugeList(archive_path: str, target_archive_path: str):
    archives = sorted(os.listdir(archive_path))
    for rain_gauge_station_file in archives:
        path = os.path.join(archive_path, rain_gauge_station_file)
        print("Start\nPath:{}".format(path))
        # head:
        # 0            product ID
        # 1            GV site
        # 2            rain gauge network
        # 3            rain gauge ID
        # 4            name of rain gauge location
        # 5            rain gauge type: tipping bucket
        # 6            resolution: 1 minute
        # 7            latitude in degree
        # 8            longitude in degree
        # 9            primary radar
        # 10           range, i.e. distance between primary radar and gauge
        # 11           azimuth between primary radar and gauge
        #              Azimuth is a direction in terms of the 360 degree compass;
        #              north at 0, east at 90, south at 180, west at 270, etc.
        # 12           gauge coordinate in pixel at X axis
        # 13           gauge coordinate in pixel at Y axis
        #              Radar location is at center pixel (75,75);
        #              Pixel (0,0) is at the SW corner. One pixel is 2kmX2km.
        # 14           radar elevation above sea level. (-99.9 not available)
        with open(path) as f:
            metadata = f.readline()
            print("Metadata for gauge station: {}".format(metadata))
            # columns_list = ["rainRate", "rainFall"]
            columns_list = ["RainRate"]
            index_list = []
            rain_rate_data_list = []
            # rain_fall_data_list = []

            # location information
            rain_gauge_latitude = float(metadata.split()[7])
            rain_gauge_longitude = float(metadata.split()[8])
            rain_gauge_range = float(metadata.split()[-5])
            rain_gauge_azimuth = float(metadata.split()[-4])
            rain_gauge_station_name = metadata.split()[2] + metadata.split()[3]
            print(rain_gauge_station_name)
            if rain_gauge_range > 100.0:
                print("{} End\nBecause the rain gauge station isn't within 100km of primary radar.\n".format(
                    rain_gauge_station_name))
                continue

            for recorde in f:
                # data:
                # 0-year 1-month 2-day 3-julian_day 4-hour 5-minute 6-second 7-rain_rate(mm/h, last minute)
                # 8-Type of interpolation for different rain events
                # 9 Rainfall bias by event
                # 10 Number of tips in a given rain event
                rain_rate = float(recorde.split()[7])
                interpolation = int(recorde.split()[8])
                # rain_fall = rain_rate / 60
                tips = int(recorde.split()[10])

                # 存在tips<30 但是 降水强度很大的情况, 待修改
                # if tips < 30 or rain_rate < 0:
                #     continue

                if rain_rate < 0.0:
                    continue

                if interpolation != 0:
                    continue

                year = int(recorde.split()[0])
                month = int(recorde.split()[1])
                day = int(recorde.split()[2])

                hour = int(recorde.split()[4])
                minute = int(recorde.split()[5])
                second = int(recorde.split()[6])

                endTime = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
                rain_rate_data_list.append(rain_rate)
                # rain_fall_data_list.append(rain_fall)
                index_list.append(endTime)

        print("Gauge file name: {}".format(rain_gauge_station_file))
        # data_all = numpy.array([rain_rate_data_list, rain_fall_data_list]).transpose()
        data_all = numpy.array([rain_rate_data_list]).transpose()
        print("Gauge data shape: {}".format(data_all.shape))
        if data_all.size == 0:
            continue
        dataFrame = pandas.DataFrame(data=data_all, index=index_list, columns=columns_list)
        dataFrame.to_csv(os.path.join(target_archive_path, "{}.csv".format(rain_gauge_station_file.split('.')[0])))
        print("{} End\nComplete\n".format(rain_gauge_station_name))


# if __name__ == '__main__':
#     # MakeGaugeList("/media/data3/zhangjianchang/DATA/GPM/RawData/2017",
#     #               "/media/data3/zhangjianchang/DATA/GPM/ProcessedGaugeCSV")