# @Time     : 1/7/2022 8:05 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : PDF_Plot.py
# @Project  : QuantitativePrecipitationEstimation
from typing import Iterable

import pandas
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def DatasetDistribution(csvFilePaths: Iterable[str]):
    dataFrameList = []
    for path in csvFilePaths:
        singleDataFrame = pandas.read_csv(path, index_col=0)
        dataFrameList.append(singleDataFrame)

    dataFrameAll = pandas.concat(dataFrameList, axis=0, ignore_index=True)
    meanRainRate = dataFrameAll["MEAN_OF_RAIN_RATE"]

    font_title_label = {"family": "Times New Roman",
                        "weight": "bold",
                        "size": 14,
                        "color": "black",
                        "style": "normal"}

    cuts = pandas.cut(meanRainRate, [0, 20, 40, 60, 80, 100, 120, 200])
    print(cuts.value_counts())
    counts = pandas.value_counts(cuts, sort=False)

    barPlot = plt.bar(counts.index.astype(str), counts)

    plt.bar_label(barPlot, counts)

    plt.yticks(fontproperties='Times New Roman', size=10, weight="regular")
    plt.xticks(fontproperties='Times New Roman', size=9, weight="regular")

    plt.ylabel("Count", fontdict=font_title_label)
    plt.xlabel("Rain Rate Interval (mm/h)", fontdict=font_title_label)
    plt.title("Dataset Distribution", fontdict=font_title_label)

    plt.savefig(
        "/media/data3/zhangjianchang/DATA/Pictures/Dataset_PDF.png",
        dpi=600,
        format='png',
    )


if __name__ == '__main__':
    if __name__ == '__main__':
        DatasetDistribution(
            ["/media/data3/zhangjianchang/DATA/Dataset/CSV/2017.csv",
             "/media/data3/zhangjianchang/DATA/Dataset/CSV/2018.csv",
             "/media/data3/zhangjianchang/DATA/Dataset/CSV/2019.csv",
             "/media/data3/zhangjianchang/DATA/Dataset/CSV/2020.csv"]
        )
