# @Time     : 1/7/2022 10:03 AM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : GaugeStationPlot.py
# @Project  : QuantitativePrecipitationEstimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")

out100kmGaugeStationLocation = {'SFL0002': (26.08333, -80.68444),
                                'SFL0003': (26.3025, -81.68833),
                                'SFL0009': (26.51278, -80.98194),
                                'SFL0015': (26.65694, -80.62972),
                                'SFL0017': (26.735, -80.89528),
                                'SFL0019': (26.27278, -81.77972),
                                'SFL0020': (26.12972, -81.7625),
                                'SFL0021': (25.99056, -81.59139),
                                'SFL0023': (25.97861, -81.48083),
                                'SFL0026': (26.85889, -80.48417),
                                'SFL0027': (26.55833, -80.70917),
                                'SFL0030': (26.43639, -80.615),
                                'SFL0045': (26.22833, -81.63194),
                                'SFL0047': (26.39306, -81.40694),
                                'SFL0049': (27.02861, -80.16528),
                                'SFL0053': (27.13889, -80.78778),
                                'SFL0054': (26.95972, -80.97694),
                                'SFL0055': (26.8225, -80.78278),
                                'SFL0059': (26.60722, -81.64972),
                                'SFL0060': (26.49889, -80.22222),
                                'SFL0061': (26.90167, -80.78889),
                                'SFL0063': (26.68194, -80.80611),
                                'SFL0064': (25.82694, -80.34417),
                                'SFL0070': (26.92444, -81.31389),
                                'SFL0074': (26.33194, -80.87972),
                                'SFL0080': (26.17194, -80.82722),
                                'SFL0081': (25.61083, -80.50972),
                                'SFL0082': (25.76139, -80.49667),
                                'SFL0092': (26.78972, -81.30278),
                                'SFL0099': (26.23083, -81.13028),
                                'SFL0100': (26.665, -80.70111),
                                'SFL0106': (26.94667, -81.56611),
                                'SFL0111': (26.51056, -80.31028),
                                'SFL0112': (27.21972, -80.465),
                                'SFL0114': (25.98972, -80.83611),
                                'SFL0118': (26.66778, -80.94889),
                                'SFL0122': (26.09444, -80.23028),
                                'SFL0127': (26.12917, -80.36556),
                                'SFL0128': (27.12222, -80.89583),
                                'SFL0129': (26.97917, -81.09),
                                'SFL0130': (27.08639, -80.66111),
                                'SFL0131': (26.98889, -80.60444),
                                'SFL0132': (27.21056, -80.91833),
                                'SFL0133': (26.64472, -80.055),
                                'SFL0134': (25.33056, -80.525),
                                'SFL0135': (26.28389, -80.96778),
                                'SFL0136': (27.19194, -80.7625),
                                'SFL0137': (25.46278, -80.3475),
                                'SFL0138': (25.92833, -80.15083),
                                'SFL0139': (25.96194, -80.26444),
                                'SFL0140': (26.70028, -80.71611),
                                'SFL0143': (26.86389, -80.63194),
                                'SFL0144': (26.69889, -80.80722),
                                'SFL0146': (26.68444, -80.3675),
                                'SFL0148': (27.33, -81.25417),
                                'SFL0149': (26.64278, -80.58083),
                                'SFL0150': (27.11861, -81.15722),
                                'SFL0151': (26.33583, -80.53667),
                                'SFL0152': (27.27278, -81.20194),
                                'SFL0153': (27.27139, -81.19222),
                                'SFL0154': (27.21611, -80.97333),
                                'SFL0155': (26.33222, -80.77417),
                                'SFL0160': (26.27861, -80.605),
                                'SFL0161': (26.28556, -80.77583),
                                'SFL0162': (25.85333, -80.76917),
                                'SFL0170': (25.76222, -80.68139),
                                'SFL0174': (26.67889, -80.53778),
                                'SFL0201': (25.85172, -80.76625),
                                'SFL0202': (27.16142, -80.43261),
                                'SFL0203': (27.12025, -80.43211),
                                'SFL0206': (26.03958, -81.02711),
                                'SFL0207': (26.05658, -81.15594),
                                'SFL0208': (26.20494, -81.16847),
                                'SFL0209': (26.20656, -80.98361),
                                'SFL0210': (25.79278, -81.2025),
                                'SFL0211': (25.70611, -80.935),
                                'SFL0212': (25.95758, -81.10367),
                                'SFL0213': (25.77872, -80.912),
                                'SFL0216': (26.872, -80.24506),
                                'SFL0217': (26.27319, -81.71725),
                                'SFL0219': (26.9195, -81.12172),
                                'SFL0222': (26.65406, -80.06825),
                                'SFL0223': (26.14353, -81.35033),
                                'SFL0225': (26.42067, -80.51756),
                                'SFL0226': (26.32786, -80.13089),
                                'SFL0227': (26.23119, -80.12422),
                                'SFL0234': (26.608, -80.94936),
                                'SFL0236': (25.93194, -81.71197),
                                'SFL0249': (26.05025, -81.70047),
                                'SFL0250': (25.61039, -80.30783),
                                'SFL0251': (26.16425, -80.29756),
                                'SFL0252': (27.02978, -81.00144),
                                'SFL0253': (27.20617, -80.80089),
                                'SFL0254': (26.06619, -80.20867),
                                'SFL0255': (26.17203, -80.82728),
                                'SFL0256': (25.54261, -80.4095),
                                'SFL0257': (25.50261, -80.46339),
                                'SFL0258': (26.76256, -80.92283),
                                'SFL0260': (25.40289, -80.55839),
                                'SFL0261': (25.47372, -80.4145),
                                'SFL0262': (25.48928, -80.34728),
                                'SFL0263': (25.51928, -80.34617),
                                'SFL0264': (25.54319, -80.33094),
                                'SFL0265': (25.80817, -80.26089),
                                'SFL0266': (25.84869, -80.18894),
                                'SFL0267': (25.91342, -80.29311),
                                'SFL0268': (25.95675, -80.43144),
                                'SFL0269': (25.61094, -80.50978),
                                'SFL0270': (25.76178, -80.50228),
                                'SFL0271': (25.77622, -80.48283),
                                'SFL0272': (25.66067, -80.48033),
                                'SFL0273': (26.13564, -80.19089),
                                'SFL0274': (26.17342, -80.17839),
                                'SFL0275': (26.20592, -80.13228),
                                'SFL0276': (26.22397, -80.17089),
                                'SFL0277': (26.22981, -80.29839),
                                'SFL0278': (26.35619, -80.29756),
                                'SFL0279': (26.41869, -80.07419),
                                'SFL0280': (26.53119, -80.05919),
                                'SFL0281': (26.81672, -80.08169),
                                'SFL0282': (26.93422, -80.14169),
                                'SFL0283': (26.85811, -81.13894),
                                'SFL0285': (26.47231, -80.44561),
                                'SFL0286': (27.03386, -81.07089),
                                'SFL0287': (27.09156, -81.00669),
                                'SFL0288': (27.19172, -81.12728),
                                'SFL0289': (27.19189, -81.128),
                                'SFL0290': (26.48481, -80.65311),
                                'SFL0291': (27.20533, -80.34061),
                                'SFL0292': (26.06147, -80.44144),
                                'SFL0293': (27.45831, -81.35431),
                                'SFL0294': (26.14536, -81.57564),
                                'SFL0297': (27.08058, -81.33631),
                                'SFL0298': (26.76478, -80.49867),
                                'SFL0299': (26.68861, -80.18806),
                                'SFL0305': (25.714, -81.02175),
                                'SFL0306': (26.04453, -81.29981),
                                'SFL0310': (26.30169, -81.43136)}

in100kmGaugeStationLocation = {'SFL0010': (28.19861, -81.23972),
                               'SFL0011': (27.32028, -80.84139),
                               'SFL0012': (27.41139, -80.92111),
                               'SFL0024': (27.26972, -80.70528),
                               'SFL0038': (28.15556, -81.115),
                               'SFL0039': (27.31389, -80.94694),
                               'SFL0041': (27.36694, -80.51417),
                               'SFL0052': (27.5025, -81.19528),
                               'SFL0065': (27.35333, -80.81611),
                               'SFL0067': (27.25389, -80.78722),
                               'SFL0068': (27.32194, -80.77528),
                               'SFL0085': (28.14, -81.35139),
                               'SFL0097': (28.37722, -81.45083),
                               'SFL0101': (27.29028, -80.25361),
                               'SFL0109': (28.04833, -81.39944),
                               'SFL0115': (27.33056, -80.46306),
                               'SFL0116': (27.35778, -80.62972),
                               'SFL0145': (27.26167, -80.35889),
                               'SFL0157': (27.47056, -80.47167),
                               'SFL0158': (28.17361, -81.31222),
                               'SFL0167': (28.15494, -81.42433),
                               'SFL0171': (27.31417, -81.02222),
                               'SFL0172': (27.68583, -81.18639),
                               'SFL0204': (27.63169, -81.26478),
                               'SFL0205': (27.40364, -81.01144),
                               'SFL0214': (28.45278, -81.17811),
                               'SFL0218': (28.03864, -81.46258),
                               'SFL0221': (27.75281, -81.07728),
                               'SFL0228': (27.49475, -80.9295),
                               'SFL0230': (27.78781, -81.32672),
                               'SFL0231': (28.25594, -81.50381),
                               'SFL0232': (27.88892, -81.01811),
                               'SFL0235': (27.59142, -81.43533),
                               'SFL0237': (27.68364, -81.02367),
                               'SFL0238': (27.54142, -81.10033),
                               'SFL0239': (27.43864, -81.20644),
                               'SFL0244': (27.54947, -81.02339),
                               'SFL0248': (27.55789, -80.82736),
                               'SFL0284': (27.80306, -81.19828),
                               'SFL0295': (27.97169, -81.41756),
                               'SFL0308': (27.65939, -81.13294),
                               'SFL0309': (27.39694, -80.42511)}


def create_map(start_latitude: float = 23.0,
               end_latitude: float = 50.0,
               start_longitude: float = -126.0,
               end_longitude: float = -66.0):
    font_title_label = {"family": "Times New Roman",
                        "weight": "bold",
                        "size": 13,
                        "color": "black",
                        "style": "normal"}

    out_longitude = [lon for name, (lat, lon) in out100kmGaugeStationLocation.items()]
    out_latitude = [lat for name, (lat, lon) in out100kmGaugeStationLocation.items()]
    out_name = [name for name, (lat, lon) in out100kmGaugeStationLocation.items()]

    in_longitude = [lon for name, (lat, lon) in in100kmGaugeStationLocation.items()]
    in_latitude = [lat for name, (lat, lon) in in100kmGaugeStationLocation.items()]
    in_name = [name for name, (lat, lon) in in100kmGaugeStationLocation.items()]

    extent = [start_longitude, end_longitude, start_latitude, end_latitude]

    cities = "/media/data3/zhangjianchang/DATA/Geography/USShp/US_Cities.shp"
    states_boundary = "/media/data3/zhangjianchang/DATA/Geography/USShp/US_StatesBoundary.shp"
    usa_boundary = "/media/data3/zhangjianchang/DATA/Geography/USShp/USA_Boundary.shp"

    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(5, 3), dpi=500)

    ax = fig.subplots(1, 1, subplot_kw={'projection': projection})

    reader_cities = Reader(cities)
    reader_states = Reader(states_boundary)
    reader_boundary = Reader(usa_boundary)

    cities = cfeature.ShapelyFeature(reader_cities.geometries(), projection, edgecolor="k", facecolor="none")
    states = cfeature.ShapelyFeature(reader_states.geometries(), projection, edgecolor="k", facecolor="none")
    boundary = cfeature.ShapelyFeature(reader_boundary.geometries(), projection, edgecolor="k", facecolor="none")

    ax.add_feature(cities, linewidth=0.5)
    ax.add_feature(states, linewidth=0.5)
    ax.add_feature(boundary, linewidth=0.5)
    ax.set_extent(extent, crs=projection)

    image_1 = Image.open("/media/data3/zhangjianchang/DATA/Geography/srtm_20_07/srtm_20_07.tif")
    image_2 = Image.open("/media/data3/zhangjianchang/DATA/Geography/srtm_20_08/srtm_20_08.tif")
    image_3 = Image.open("/media/data3/zhangjianchang/DATA/Geography/srtm_21_07/srtm_21_07.tif")
    image_4 = Image.open("/media/data3/zhangjianchang/DATA/Geography/srtm_21_08/srtm_21_08.tif")

    image_array1 = np.array(image_1)
    image_array2 = np.array(image_2)
    image_array3 = np.array(image_3)
    image_array4 = np.array(image_4)

    map_1 = np.ma.concatenate([image_array1, image_array3], axis=1)
    map_2 = np.ma.concatenate([image_array2, image_array4], axis=1)
    Map = np.ma.concatenate([map_1, map_2], axis=0)
    Map = np.flip(Map, axis=0)

    colorMap = ListedColormap(
        ['#001F99', '#4469FF', "#335B27", "#729D67",
         "#409C37", "#3DBF29", "#77FF77", "#C9FA2F",
         "#F5FC94", "#FFF33A", "#E0DB1F",
         "#C8C633", "#DD8D2E", "#885F22"]
    )

    longitude = np.linspace(-85, -75, 12000)
    latitude = np.linspace(20, 30, 12000)

    im = ax.pcolormesh(longitude, latitude, Map, cmap=colorMap, transform=projection, vmin=-20, vmax=120)

    ax.plot(out_longitude, out_latitude,
            marker='.', color='red', markersize=2, label="distance > 100km", linestyle='', transform=projection)

    ax.plot(in_longitude, in_latitude,
            marker='.', color='black', markersize=2, label="distance <= 100km", linestyle='', transform=projection)

    ax.plot(-80.65408325, 28.11319351,
            marker='.', color='yellow', markersize=4, label="KMLB", linestyle='', transform=projection)

    ax.axis((start_longitude, end_longitude,
             start_latitude, end_latitude))
    x = ax.set_xticks([-83, -82, -81, -80, -79], crs=projection)
    y = ax.set_yticks([25, 26, 27, 28, 29, 30], crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    color_bar = fig.colorbar(im)
    color_bar.ax.tick_params(labelsize=7)

    color_bar.set_label("Elevation", fontdict=font_title_label)
    color_bar.set_ticks([-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    color_bar.set_ticklabels([-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    color_bar.ax.set_title("m", fontdict=font_title_label)

    plt.yticks(fontproperties='Times New Roman', size=7, weight="regular")
    plt.xticks(fontproperties='Times New Roman', size=7, weight="regular")

    plt.ylabel("Latitude (deg.)", fontdict=font_title_label)
    plt.xlabel("Longitude (deg.)", fontdict=font_title_label)
    plt.tick_params(labelsize=8)
    plt.title("Study Domain", fontdict=font_title_label)
    return ax


if __name__ == '__main__':
    a = create_map(start_latitude=25, end_latitude=30, start_longitude=-83.5, end_longitude=-78.5)
    plt.savefig(
        "/media/data3/zhangjianchang/DATA/Pictures/Gauge_MAP.png",
        dpi=600,
        format='png',
        bbox_inches='tight'
    )
