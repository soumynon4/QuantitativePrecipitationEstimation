# @Time     : 1/11/2022 1:11 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : DomainPlotForEstimation.py
# @Project  : QuantitativePrecipitationEstimation
import numpy
import pyart
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _mask_outside(flag, data, v1, v2):
    if flag:
        data = numpy.ma.masked_invalid(data)
        data = numpy.ma.masked_outside(data, v1, v2)
    return data


def EstimationPlotDomain(figure, subFigure, radar_range_data, radar_azimuth_data, radar_elevation_data,
                         estimation: numpy.ndarray, mask_outside=False):
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

    precipitationUnit = "mm"
    minValue = 0
    maxValue = 30
    precipitationColorMap = ListedColormap(['#00A1F7', '#00EDED', '#00D900', '#009100',
                                            '#FFFF00', '#E7C100', '#FF9100', '#FF0000',
                                            '#D70000', '#C10000', '#FF00F1', '#AD91F1'])
    ticks = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]

    data = numpy.ma.array(estimation, mask=(estimation <= 0.0))

    x, y, _ = pyart.core.antenna_vectors_to_cartesian(radar_range_data,
                                                      radar_azimuth_data,
                                                      radar_elevation_data,
                                                      edges=True)
    x = x / 1000.0
    y = y / 1000.0

    data = _mask_outside(mask_outside, data, minValue, maxValue)
    pm = subFigure.pcolormesh(x, y, data, vmin=minValue, vmax=maxValue, cmap=precipitationColorMap)

    subFigure.set_title("Precipitation", fontdict=titleFontDict)
    subFigure.set_xlabel("East West distance from radar (km)", fontdict=titleFontDict)
    subFigure.set_ylabel("North South distance from radar (km)", fontdict=titleFontDict)

    subFigure.set_yticks(numpy.arange(-100, 110, 25))
    subFigure.set_xticks(numpy.arange(-100, 110, 25))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

    colorBar = figure.colorbar(pm)
    colorBar.ax.tick_params(labelsize=9)

    colorBar.set_label("Precipitation (mm)", fontdict=colorFontDict)
    colorBar.set_ticks(ticks)
    colorBar.set_ticklabels(ticks)
    colorBar.ax.set_title(precipitationUnit, fontdict=colorFontDict)

    return figure
