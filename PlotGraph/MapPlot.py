# @Time     : 7/19/2021 6:50 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : MapPlot.py
# @Project  : QuantitativePrecipitationEstimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader
from matplotlib.colors import ListedColormap
from matplotlib.image import imread
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.pyplot import MultipleLocator


def create_map(start_longitude: float = -126.0,
               end_longitude: float = -66.0,
               start_latitude: float = 23.0,
               end_latitude: float = 50.0):

    font_title_label = {"family": "Arial",
                        "weight": "bold",
                        "size": 14,
                        "color": "black",
                        "style": "normal"}
    font_ticks = {"family": "Arial",
                  "weight": "bold",
                  "size": 7,
                  "color": "black",
                  "style": "normal"}
    colormap = ListedColormap(["#01a0f6", "#00ecec", "#00d800", "#019000", "#ffff00", "#e7c000",
                               "#ff9000", "#ff0000", "#d60000", "#c00000", "#ff00f0", "#9600b4",
                               "#ad90f0"])

    # 1. 选定可视化区域范围经纬度
    # extent = [-130, -70, 25, 50]  美国本土起始截止经纬度
    # [起始经度, 截至经度, 起始纬度, 截至纬度]

    extent = [start_longitude, end_longitude, start_latitude, end_latitude]

    cities = "c:\\users\\admin\\desktop\\USShp\\US_Cities.shp"
    states_boundary = "c:\\users\\admin\\desktop\\USShp\\US_StatesBoundary.shp"
    usa_boundary = "c:\\users\\admin\\desktop\\USShp\\USA_Boundary.shp"
    # shape_file path

    projection = ccrs.PlateCarree()
    # 2.选择投影方式, 创建坐标系

    fig = plt.figure(figsize=(5, 3), dpi=500)
    # 3.创建画板

    ax = fig.subplots(1, 1, subplot_kw={'projection': projection})
    # 4.创建子图, 将投影传入

    reader_cities = Reader(cities)
    reader_states = Reader(states_boundary)
    reader_boundary = Reader(usa_boundary)
    # 读取shape_file文件

    cities = cfeature.ShapelyFeature(reader_cities.geometries(), projection, edgecolor="k", facecolor="none")
    states = cfeature.ShapelyFeature(reader_states.geometries(), projection, edgecolor="k", facecolor="none")
    boundary = cfeature.ShapelyFeature(reader_boundary.geometries(), projection, edgecolor="k", facecolor="none")
    # 5.读取shape_file文件中的地理信息

    ax.add_feature(cities, linewidth=0.5)
    ax.add_feature(states, linewidth=0.5)
    ax.add_feature(boundary, linewidth=0.5)

    ax.set_extent(extent, crs=projection)
    # 6.向子图中添加shape_file中的地理信息, 限定子图的区域范围

    fname = "c:\\users\\admin\\desktop\\NE1_50M_SR_W\\NE1_50M_SR_W.tif"
    img = imread(fname)
    # 7.选择高分辨率自然地形光栅数据地图, 可在natural_earth下载

    ax.imshow(img, origin="upper", transform=projection, extent=[-180, 180, -90, 90])
    # 8.读取地图数据, 经过投影和限定区域(全球地图即[-180, 180, -90, 90])

    # grid_line = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="k", alpha=1,linestyle="--")
    # 9.设置网格线格式

    # x1, y1 = [-98.3, -98.3, -96.3, -96.3, -98.3], [33.58, 31.58, 31.58, 33.58, 33.58]
    # x2, y2 = [-124.0, -124.0, -120.0, -120.0, -124.0], [40.0, 36.0, 36.0, 40.0, 40.0]
    # ax.plot(x1, y1, linewidth=0.8, color="red")  # 设置形状的线的属性
    # ax.fill(x1, y1, color='red', alpha=0.4)  # 设置填充色的属性
    # ax.plot(x2, y2, linewidth=0.8, color="red")  # 设置所绘的形状的线的属性
    # ax.fill(x2, y2, color='blue', alpha=0.4)  # 设置形状内填充色的属性
    # 10.在子图中根据经纬度画出形状

    # grid_line.xlabels_top = False  # 关闭顶端的经纬度标签
    # grid_line.ylabels_right = False  # 关闭右侧的经纬度标签

    # grid_line.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    # grid_line.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式

    # grid_line.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + 10, 10))
    # grid_line.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + 10, 10))

    # 添加雷达数据

    # data = np.load("c:\\users\\admin\\desktop\\image_KFWS20190424_004602.npy")
    # data = np.ma.array(data, mask=(data==0))
    # if len(data.shape) == 2:
    #     lon = np.linspace(-97, -96,num=data.shape[1])
    #     lat = np.linspace(31, 32,num=data.shape[0])
    # else:
    #     lon = np.linspace(-97, -96,num=data.shape[2])
    #     lat = np.linspace(31, 32,num=data.shape[1])
    #     data = data[0, ...]
    # im = ax.pcolormesh(lon, lat, data, transform=projection,cmap=colormap, vmin=0, vmax=65)
    # color_bar = fig.colorbar(im)
    # color_bar.ax.tick_params(labelsize=9)
    # color_bar.set_ticks([i for i in range(0, 70, 5)])
    # color_bar.set_ticklabels([i for i in range(0, 70, 5)])
    # color_bar.set_label("Reflectivity", fontdict=font_ticks)
    # color_bar.ax.set_title("dBZ", fontdict=font_ticks)
    # 添加雷达数据
    # x_major_locator = MultipleLocator(5)
    # y_major_locator = MultipleLocator(1)
    #
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    # x = ax.set_xticks([-126, -114, -102, -90, -78, -66], crs=ccrs.PlateCarree())  # 子图设置x轴坐标
    # y = ax.set_yticks([23, 32, 41, 50], crs=ccrs.PlateCarree())  # 子图设置y轴坐标
    x = ax.set_xticks([-120, -110, -100, -90, -80, -70], crs=ccrs.PlateCarree())  # 子图设置x轴坐标
    y = ax.set_yticks([30, 40, 50], crs=ccrs.PlateCarree())  # 子图设置y轴坐标
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    plt.yticks(fontproperties='Arial', size=10, weight="regular")
    plt.xticks(fontproperties='Arial', size=10, weight="regular")

    # plt.ylabel("Latitude (deg.)", fontdict=font_title_label)
    # plt.xlabel("Longitude (deg.)", fontdict=font_title_label)
    # plt.tick_params(labelsize=8)
    # plt.title("Study Domain", fontdict=font_title_label)  # 设置子图的title
    return ax


if __name__ == '__main__':
    a = create_map()
    # plt.show()
    plt.savefig("c:\\users\\admin\\desktop\\DOMAIN_MAP.png", dpi=600, pad_inches=0, format='png')