# @Time     : 7/19/2021 6:48 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : TifPlot.py
# @Project  : QuantitativePrecipitationEstimation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.use("Agg")


def Tif_Plot(file_path, data_lon_lat, plot_lon_lat):
    filename = file_path.split("/")[-1].split(".")[0]
    # 1.读取
    # PIL读取tif地形图文件
    img = Image.open("{}".format(file_path))

    # 将PIL.Image对象转为numpy.array
    img_array = np.array(img)

    # 查看array大小
    # print(img_array.shape)

    # 将无效值遮盖
    img_array = np.ma.MaskedArray(img_array, mask=(img_array < -30))

    # 设定数据的经纬度
    longitude = np.linspace(data_lon_lat[0], data_lon_lat[1], 6000)
    latitude = np.linspace(data_lon_lat[2], data_lon_lat[3], 6000)

    # 2.画图
    # 创建画布
    fig = plt.figure(figsize=(6, 6))

    # 在画布中添加子图
    ax = fig.add_subplot(1, 1, 1)

    # 设置子图和坐标轴label
    font_title_label = {"family": "Arial",  # 字体: Times New Roman
                        "weight": "normal",
                        "size": 20,
                        "color": "black",
                        "style": "normal"}  # italic 斜体
    # plt.title("Study Domain of State of Texas", fontdict=font_title_label)
    # plt.xlabel("Longitude (deg.)", fontdict=font_title_label)
    # plt.ylabel("Latitude (deg.)", fontdict=font_title_label)

    # 设置x, y轴范围
    ax.axis((plot_lon_lat[0], plot_lon_lat[1], plot_lon_lat[2],
             plot_lon_lat[3]))  # 通过传入[x_min, x_max, y_min, y_max]对应的值, 限定画图的x轴和y轴的最大最小值
    # plt.xticks([-98.3, -97.8, -97.3, -96.8, -96.3], fontsize=9)
    # plt.yticks([31.58, 32.08, 32.58, 33.08, 33.58], fontsize=9)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)

    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # ax.set_xticklabels(["125°W", "124°W", "123°W", "122°W", "121°W", "120°W"], fontproperties='Arial', size=14)
    # ax.set_yticklabels(["35°N", "36°N", "37°N", "38°N", "39°N", "40°N"], fontproperties='Arial', size=14)
    ax.set_xticklabels(["100°W","98°W", "97°W", "96°W", "95°W"], fontproperties='Arial', fontsize=20)
    ax.set_yticklabels(["30°N","31°N", "32°N", "33°N", "34°N", "35°N"], fontproperties='Arial', fontsize=20)
    # 通过plt.xticks()和plt.yticks() 可以设置x轴和y轴的刻度显示

    # 用pcolormesh对二维矩阵进行画图, pcolormesh画图时, 图片左下角为矩阵原点
    cmap = ListedColormap(
        ["#335B27", "#729D67", "#409C37", "#3DBF29", "#77FF77", "#C9FA2F", "#F5FC94", "#FFF33A", "#E0DB1F",
         "#C8C633", "#DD8D2E", "#885F22"])
    img_array = np.flip(img_array, axis=0)
    im = ax.pcolormesh(longitude, latitude, img_array, cmap=cmap, vmin=0, vmax=1200)

    # 设置colorbar
    color_bar = fig.colorbar(im)
    color_bar.ax.tick_params(labelsize=9)  # 设置色标刻度字体大小
    font_colorbar = {"family": "Arial",
                     "color": "black",
                     "weight": "normal",
                     "size": 20}
    # color_bar.set_label("Elevation", fontdict=font_colorbar)
    color_bar.set_ticks([i for i in range(0, 1300, 100)])  # 设置colorbar刻度
    color_bar.set_ticklabels([i for i in range(0, 1300, 100)])  # 设置colorbar刻度的显示标签
    color_bar.ax.set_title("m", fontdict=font_title_label)  # 设置colorbar的title
    # plt.savefig('./{}.svg'.format(filename), bbox_inches='tight', dpi=600, pad_inches=0, format='svg')
    plt.savefig('./{}.png'.format(filename), bbox_inches='tight', dpi=600, pad_inches=0)
    plt.clf()
    ax.clear()


if __name__ == '__main__':
    Tif_Plot("/media/data3/zhangjianchang/Elevation/srtm_17_06/srtm_17_06.tif", data_lon_lat=[-100, -95, 30, 35],
             plot_lon_lat=[-98.3, -96.3, 31.58, 33.58])
    # Tif_Plot("/media/data3/zhangjianchang/Elevation/srtm_12_05/srtm_12_05.tif", data_lon_lat=[-125, -120, 35, 40],
    #          plot_lon_lat=[-124.0, -120.0, 36.0, 40.0])