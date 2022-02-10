# @Time     : 1/25/2022 7:39 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : PPI_Plot.py
# @Project  : QuantitativePrecipitationEstimation
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pyart
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")


def _mask_outside(flag, data, v1, v2):
    if flag:
        data = numpy.ma.masked_invalid(data)
        data = numpy.ma.masked_outside(data, v1, v2)
    return data


def PPI(figure, subFigure, radar_range_data, radar_azimuth_data, radar_elevation_data,
        variable: numpy.ma.MaskedArray, variable_name: str, unit_name: str,
        min_value: float, max_value: float, mask_outside=False):
    titleFontDict = {"family": "serif",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 16}

    colorFontDict = {"family": "serif",
                     "style": "normal",
                     "color": "black",
                     "weight": "bold",
                     "size": 14}

    minValue = min_value
    maxValue = max_value
    interval = (maxValue - minValue) / 14
    precipitationColorMap = ListedColormap(['#A4A4A4', '#00A1F7', '#00EDED', '#00D900', '#009100','#FFFF00', '#E7C100',
                                            '#FF9100', '#FF0000', '#D70000', '#C10000', '#FF00F1', '#AD91F1', '#3F3FBF'])
    ticks = [i * interval for i in range(15)]

    data = variable

    x, y, _ = pyart.core.antenna_vectors_to_cartesian(radar_range_data,
                                                      radar_azimuth_data,
                                                      radar_elevation_data,
                                                      edges=True)
    x = x / 1000.0
    y = y / 1000.0

    data = _mask_outside(mask_outside, data, minValue, maxValue)
    pm = subFigure.pcolormesh(x, y, data, vmin=minValue, vmax=maxValue, cmap=precipitationColorMap)

    subFigure.set_title("{}".format(variable_name.capitalize()), fontdict=titleFontDict)
    subFigure.set_xlabel("East West distance from radar (km)", fontdict=titleFontDict)
    subFigure.set_ylabel("North South distance from radar (km)", fontdict=titleFontDict)

    subFigure.set_yticks(numpy.arange(-400, 500, 100))
    subFigure.set_xticks(numpy.arange(-400, 500, 100))
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)

    colorBar = figure.colorbar(pm)
    colorBar.ax.tick_params(labelsize=9)

    colorBar.set_ticks(ticks)
    colorBar.set_ticklabels(ticks)
    colorBar.ax.set_title(unit_name, fontdict=colorFontDict)

    return figure


def RadarPPI(radarFilePath: str, needVariable: str, title: str, unit: str,
             sweep: int, min_value: float, max_value: float, save_path: str):

    radar = pyart.io.read_nexrad_archive(radarFilePath)
    print("VCP mode: {}".format(radar.metadata["vcp_pattern"]))
    print(radar.fixed_angle)
    start, end = radar.get_start_end(sweep)

    azimuths = radar.azimuth["data"][start: end + 1]
    minDifferenceIndex = numpy.argmin(numpy.abs(azimuths - 0.0))
    azimuth1 = numpy.expand_dims(azimuths[minDifferenceIndex:], axis=0)
    azimuth2 = numpy.expand_dims(azimuths[:minDifferenceIndex], axis=0)
    azimuths = numpy.concatenate([azimuth1, azimuth2], axis=1)
    azimuths = numpy.squeeze(azimuths)

    elevations = radar.elevation["data"][start: end + 1]
    elevation1 = numpy.expand_dims(elevations[minDifferenceIndex:], axis=0)
    elevation2 = numpy.expand_dims(elevations[:minDifferenceIndex], axis=0)
    elevations = numpy.concatenate([elevation1, elevation2], axis=1)
    elevations = numpy.squeeze(elevations)

    variable = radar.fields["{}".format(needVariable)]["data"][start: end + 1, ...]
    variable_1 = variable[minDifferenceIndex:, ...]
    variable_2 = variable[0:minDifferenceIndex, ...]
    variable = numpy.ma.concatenate([variable_1, variable_2], axis=0)

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    PPI(fig, ax, radar.range["data"], azimuths, elevations, variable, title, unit, min_value, max_value)
    plt.savefig(fname=save_path, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    RadarPPI("/media/data3/zhangjianchang/RadarData/2020/KMLB20200424_115539_V06",
             "reflectivity",
             "reflectivity factor",
             "dBZ",
             0,
             0.0,
             70.0,
             "/media/data3/zhangjianchang/DATA/Pictures/ReflectivityPPI_20200424_115539_V06.png")
