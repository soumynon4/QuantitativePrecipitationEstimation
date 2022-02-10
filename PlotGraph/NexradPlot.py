# @Time     : 7/10/2021 9:37 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : NexradPlot.py
# @Project  : QuantitativePrecipitationEstimation
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pyart
from matplotlib.colors import ListedColormap


def plotRadarNpy(radarDataPath: str, savingDirectory: str, colormap: list = None) -> None:
    if colormap is None:
        colormap = ListedColormap(["#01a0f6", "#00ecec", "#00d800", "#019000", "#ffff00", "#e7c000",
                                   "#ff9000", "#ff0000", "#d60000", "#c00000", "#ff00f0", "#9600b4",
                                   "#ad90f0"])
    radarData = np.load(radarDataPath)
    radarData = np.ma.MaskedArray(radarData, mask=(radarData == 0))
    if sys.platform == "linux":
        dataTime = radarDataPath.split("/")[-1].split(".")[0][4:]
    else:
        dataTime = radarDataPath.split("\\")[-1].split(".")[0][4:]

    for i in range(radarData.shape[0]):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)

        font_title_label = {"family": "serif",
                            "weight": "normal",
                            "size": 14,
                            "color": "black",
                            "style": "italic"}

        plt.title('{}-{}-{} {}:{}:{}UTC Reflectivity({})'.
                  format(dataTime[:4], dataTime[4:6], dataTime[6:8], dataTime[9:11], dataTime[11:13], dataTime[13:15],
                         i + 1),
                  fontdict=font_title_label)
        im = ax.pcolormesh(radarData[i, ...], cmap=colormap, vmin=0, vmax=65)
        color_bar = fig.colorbar(im)
        color_bar.ax.tick_params(labelsize=9)
        color_bar.set_ticks([i for i in range(0, 70, 5)])
        color_bar.set_ticklabels([i for i in range(0, 70, 5)])
        color_bar.set_label("Reflectivity", fontdict=font_title_label)
        color_bar.ax.set_title("dBZ", fontdict=font_title_label)

        if os.path.exists(savingDirectory):
            plt.savefig(os.path.join(savingDirectory, "{}_{}.png".format(dataTime, i + 1)))
            plt.clf()
            ax.clear()
        else:
            os.mkdir(savingDirectory)
            plt.savefig(os.path.join(savingDirectory, "{}_{}.png".format(dataTime, i + 1)))
            plt.clf()
            ax.clear()
        print("{}th sweep Finished.".format(i + 1))


def plotNEXRAD(fig, ax, QCPath, variable, sweep: int, cmap=None, mask_outside=False):
    titleFontDict = {"family": "serif",
                     "style": "italic",
                     "weight": "normal",
                     "color": "black",
                     "size": 16}

    colorFontDict = {"family": "serif",
                     "style": "italic",
                     "color": "black",
                     "weight": "normal",
                     "size": 14}
    units = {"reflectivity": "dBZ",
             "cross_correlation_ratio": "ratio",
             "differential_phase": "degrees",
             "KDP": "degrees/km",
             "ZDR": "dB"}
    # families=[ 'fantasy','Tahoma', 'monospace','Times New Roman']
    # styles = ['normal', 'italic', 'oblique']
    # weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

    if variable not in units.keys():
        raise ValueError(
            "Variable is not in data.options:\n-reflectivity\n-cross_correlation_ratio\n-differential_phase"
            "\n-KDP\n-ZDR")

    if cmap is None:
        cmap = getCmapDict(variable)["cmap"]

    minValue = getCmapDict(variable)["vmin"]
    maxValue = getCmapDict(variable)["vmax"]

    if variable not in units.keys():
        raise ValueError(
            "Variable is not in data.\tOptions:\n-reflectivity\n-cross_correlation_ratio\n-differential_phase"
            "\n-KDP\n-ZDR")

    #radar = drops_reader(QCPath)
    radar = pyart.io.read_nexrad_archive(QCPath)
    # radartemp = netCDF4.Dataset(QCPath)
    # start1 = radartemp.variables["ray_start_index"][:][2520]
    # start2 = radartemp.variables["ray_start_index"][:][3240]
    # data = radartemp.variables["Reflectivity"][start1:start2]
    # data = np.reshape(data, [-1, 1832])

    data = _get_data(radar, variable, sweep)
    x, y, _ = _get_x_y_z(radar, sweep)
    data = _mask_outside(mask_outside, data, minValue, maxValue)
    pm = ax.pcolormesh(x, y, data, vmin=minValue, vmax=maxValue, cmap=cmap)

    ax.set_title("BF-DROPS: {}(sweep:{})".format(variable.capitalize(), sweep), fontdict=titleFontDict)
    ax.set_xlabel("East West distance from radar (km)", fontdict=titleFontDict)
    ax.set_ylabel("North South distance from radar (km)", fontdict=titleFontDict)

    colorBar = fig.colorbar(pm)
    colorBar.ax.tick_params(labelsize=9)

    colorBar.set_label(variable.capitalize(), fontdict=colorFontDict)
    colorBar.set_ticks(getCmapDict(variable)["ticks"])
    colorBar.set_ticklabels(getCmapDict(variable)["ticks"])
    colorBar.ax.set_title(units[variable], fontdict=colorFontDict)

    return fig


def _get_data(radar, field, sweep, mask_tuple=None, filter_transitions=True, gatefilter=None):
    """ Retrieve and return data from a plot function. """
    sweep_slice = radar.get_slice(sweep)
    data = radar.fields[field]['data'][sweep_slice]

    # mask data if mask_tuple provided
    if mask_tuple is not None:
        mask_field, mask_value = mask_tuple
        mdata = radar.fields[mask_field]['data'][sweep_slice]
        data = np.ma.masked_where(mdata < mask_value, data)

    # mask data if gatefilter provided
    if gatefilter is not None:
        mask_filter = gatefilter.gate_excluded[sweep_slice]
        data = np.ma.masked_array(data, mask_filter)

    # filter out antenna transitions
    if filter_transitions and radar.antenna_transition is not None:
        in_trans = radar.antenna_transition[sweep_slice]
        data = data[in_trans == 0]

    return data


def _get_x_y_z(radar, sweep, edges=True, filter_transitions=True):
    """ Retrieve and return x, y, and z coordinate in km. """
    x, y, z = radar.get_gate_x_y_z(sweep, edges=edges, filter_transitions=filter_transitions)
    # convert to km
    x = x / 1000.0
    y = y / 1000.0
    z = z / 1000.0
    return x, y, z


def _mask_outside(flag, data, v1, v2):
    if flag:
        data = np.ma.masked_invalid(data)
        data = np.ma.masked_outside(data, v1, v2)
    return data


def getCmapDict(variable):
    cmap_dict = defaultdict(dict)
    cmap_dict["reflectivity"]["cmap"] = ListedColormap(['#000000', '#00A1F7', '#00EDED', '#00D900',
                                                        '#009100', '#FFFF00', '#E7C100', '#FF9100', '#FF0000',
                                                        '#D70000', '#C10000', '#FF00F1', '#9700B5', '#AD91F1'])
    cmap_dict["reflectivity"]["vmin"] = 0
    cmap_dict["reflectivity"]["vmax"] = 84
    cmap_dict["reflectivity"]["ticks"] = [i for i in range(0, 15 * 6, 6)]

    cmap_dict["differential_phase"]["cmap"] = ListedColormap(['#000000', '#00A1F7', '#00EDED', '#00D900',
                                                              '#009100', '#FFFF00', '#E7C100', '#FF9100', '#FF0000',
                                                              '#D70000', '#C10000', '#FF00F1', '#9700B5', '#AD91F1'])
    cmap_dict["differential_phase"]["vmin"] = 0
    cmap_dict["differential_phase"]["vmax"] = 84
    cmap_dict["differential_phase"]["ticks"] = [i for i in range(0, 15 * 6, 6)]

    cmap_dict["cross_correlation_ratio"]["cmap"] = ListedColormap(
        ['#00A1F7', '#00EDED', '#00D900', '#009100', '#E7C100',
         '#FF0000', '#D70000', '#FF00F1', '#9700B5', '#AD91F1'])
    cmap_dict["cross_correlation_ratio"]["vmin"] = 0
    cmap_dict["cross_correlation_ratio"]["vmax"] = 1
    cmap_dict["cross_correlation_ratio"]["ticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    cmap_dict["KDP"]["cmap"] = ListedColormap(['#00A1F7', '#00EDED', '#00D900', '#009100', '#E7C100',
                                               '#FF0000', '#D70000', '#FF00F1', '#9700B5', '#AD91F1'])
    cmap_dict["KDP"]["vmin"] = 0
    cmap_dict["KDP"]["vmax"] = 1
    cmap_dict["KDP"]["ticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    cmap_dict["ZDR"]["cmap"] = ListedColormap(['#00A1F7', '#00EDED', '#00D900', '#009100', '#E7C100',
                                               '#FF0000', '#D70000', '#FF00F1', '#9700B5', '#AD91F1'])
    cmap_dict["ZDR"]["vmin"] = 0
    cmap_dict["ZDR"]["vmax"] = 1
    cmap_dict["ZDR"]["ticks"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return cmap_dict[variable]


if __name__ == '__main__':
    # matplotlib.use("Agg")

    # Function 1
    # plotRadarNpy("c:\\users\\admin\\desktop\\KFWS20190508_150720.npy")

    # Function 2
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    plotNEXRAD(fig, ax, "c:\\users\\admin\\desktop\\KMUX20210127_084537_V06.V06", "reflectivity", sweep=0)
    plt.show()

