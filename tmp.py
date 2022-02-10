# @Time     : 12/6/2021 9:18 AM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : tmp.py
# @Project  : QuantitativePrecipitationEstimation
import os
import pyart

path = "/media/data3/zhangjianchang/RadarData/2020"
files = os.listdir(path)

for number in range(len(files)):
    radar = pyart.io.read_nexrad_archive(os.path.join(path, files[number]))
    print(radar.time["data"][-1])
    # print(radar.metadata)
    print(radar.fixed_angle)
    print(radar.fixed_angle["data"].shape)
    print(radar.metadata["vcp_pattern"])
    if number == 50:
        break
    print("-"*20)

