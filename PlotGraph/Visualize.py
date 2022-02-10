# @Time     : 7/10/2021 9:21 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : Visualize.py
# @Project  : QuantitativePrecipitationEstimation
import sys
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from scipy.stats.stats import pearsonr

from tools.Metric import NSE, RMSE, BIAS, MAE

matplotlib.use("Agg")
plt.rcParams["font.weight"] = "bold"


def DensityPlot(radar_estimation: np.ndarray, gauge_label: np.ndarray, path: str, maxValue=60, minValue=0):
    density_color_bar = LinearSegmentedColormap.from_list(
        'DENSITY', ['darkblue', 'blue', 'deepskyblue', 'cyan', 'springgreen',
                    'yellow', 'orange', 'darkorange', 'red', 'firebrick']
    )

    radar_estimation = np.squeeze(radar_estimation)
    gauge_label = np.squeeze(gauge_label)

    bias = BIAS(radar_estimation, gauge_label)
    corr = pearsonr(radar_estimation, gauge_label)[0]
    nse = NSE(radar_estimation, gauge_label)
    rootMeanSquareError = RMSE(radar_estimation, gauge_label)
    mae = MAE(radar_estimation, gauge_label)
    estimation_mean = radar_estimation.mean()
    label_mean = gauge_label.mean()

    titleFontDict = {"family": "DejaVu Sans",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 25}

    textFontDict = {"family": "DejaVu Sans",
                    "style": "normal",
                    "weight": "bold",
                    "color": "black",
                    "size": 25}

    ticksFontDict = {"family": "DejaVu Sans",
                     "style": "normal",
                     "weight": "bold",
                     "color": "black",
                     "size": 20}

    fig = plt.figure(figsize=(18, 15), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    x = range(int(maxValue))
    ax.plot(x, x, color='red', lw=2.0)

    density = np.zeros([int(maxValue), int(maxValue)])

    for i in range(radar_estimation.shape[0]):
        index_gauge = int(gauge_label[i].item())
        index_estimation = int(radar_estimation[i].item())

        if index_gauge > int(maxValue - 1):
            index_gauge = int(maxValue - 1)
        if index_gauge < 0:
            index_gauge = 0

        if index_estimation >= int(maxValue - 1):
            index_estimation = int(maxValue - 1)
        if index_estimation < 0:
            index_estimation = 0

        density[index_estimation, index_gauge] += 1

    density[density == 0] = np.nan
    image_data = np.ma.masked_invalid(density)

    im = ax.pcolormesh(np.ma.log10(image_data), cmap=density_color_bar, vmin=0, vmax=2)
    color_bar = fig.colorbar(im)
    color_bar.ax.tick_params(labelsize=23)
    color_bar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    color_bar.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    color_bar.set_label("Log (bin counts)", fontdict=titleFontDict)
    # color_bar.ax.set_title("Counts", fontdict=titleFontDict)
    plt.xlim([minValue, maxValue])
    plt.ylim([minValue, maxValue])

    plt.text(2, int(maxValue - 5), "Mean of Gauge (mm/h) = {:.2f}".format(label_mean), fontdict=textFontDict)
    plt.text(2, int(maxValue - 10), "Mean of Estimation (mm/h) = {:.2f}".format(estimation_mean), fontdict=textFontDict)
    # plt.text(2, int(maxValue - 15), "Bias = {:.2f}".format(bias), fontdict=textFontDict)
    plt.text(2, int(maxValue - 15), "Corr. = {:.2f}".format(corr), fontdict=textFontDict)
    # plt.text(2, int(maxValue - 25), "NSE = {:.2f}".format(nse), fontdict=textFontDict)
    plt.text(2, int(maxValue - 20), "RMSE = {:.2f}".format(rootMeanSquareError), fontdict=textFontDict)
    plt.text(2, int(maxValue - 25), "MAE = {:.2f}".format(mae), fontdict=textFontDict)

    ax.set_title("Estimation vs Ground (Hourly)", fontdict=titleFontDict)
    ax.set_xlabel("Ground Gauge (mm)", fontdict=titleFontDict)
    ax.set_ylabel("Rain rate Estimation (mm)", fontdict=titleFontDict)

    ax.set_yticks([20 * i for i in range(int(maxValue // 20) + 1)])
    ax.set_yticklabels([20 * i for i in range(int(maxValue // 20) + 1)], fontdict=ticksFontDict)
    ax.set_xticks([20 * i for i in range(int(maxValue // 20) + 1)])
    ax.set_xticklabels([20 * i for i in range(int(maxValue // 20) + 1)], fontdict=ticksFontDict)

    plt.savefig(fname=path, bbox_inches="tight", pad_inches=0, format="png")


def PlotField(path, save=True):
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
    # units = {"reflectivity": "dBZ",
    #          "cross_correlation_ratio": "ratio",
    #          "differential_phase": "degrees",
    #          "KDP": "degrees/km",
    #          "ZDR": "dB"}

    field = np.load(path)[0, 0, ...]
    colorMap = ListedColormap(['#000000', '#00A1F7', '#00EDED', '#00D900',
                               '#009100', '#FFFF00', '#E7C100', '#FF9100', '#FF0000',
                               '#D70000', '#C10000', '#FF00F1', '#9700B5', '#AD91F1'])
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(401)
    y = np.arange(401)

    pm = ax.pcolormesh(x, y, field, vmin=0, vmax=70, cmap=colorMap)

    ax.set_title("Reflectivity", fontdict=titleFontDict)
    # ax.set_xlabel("East West distance from radar (km)", fontdict=titleFontDict)
    # ax.set_ylabel("North South distance from radar (km)", fontdict=titleFontDict)

    colorBar = fig.colorbar(pm)
    colorBar.ax.tick_params(labelsize=9)

    colorBar.set_label("dBZ", fontdict=colorFontDict)
    colorBar.set_ticks(np.arange(15) * 5)
    colorBar.set_ticklabels(np.arange(15) * 5)
    # colorBar.ax.set_title("dBZ", fontdict=colorFontDict)
    if save:
        plt.savefig("Filed.png")
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    model = "QPET"
    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_EstimationHourly.pkl".format(
                model, model
            ), "rb") as handle1:
        pred = np.array(pickle.load(handle1))

    with open(
            "/media/data3/zhangjianchang/DATA/Estimation/{}/{}_Test_LabelHourly.pkl".format(
                model, model
            ), "rb") as handle2:
        label = np.array(pickle.load(handle2))

    DensityPlot(
        pred, label,
        path="/media/data3/zhangjianchang/DATA/Pictures/{}/{}_DensityPlot.png".format(model, model))
