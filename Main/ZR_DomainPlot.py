# @Time     : 1/6/2022 11:29 AM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : ZR_DomainPlot.py
# @Project  : QuantitativePrecipitationEstimation
import os

import numpy
import pyart
import matplotlib
import matplotlib.pyplot as plt

from Methods.ReflectivityRainrateRelation import ma_QPEbyZhAndZdr, ma_QPEbyZh
from Kit.DomainPlotForEstimation import EstimationPlotDomain

matplotlib.use("Agg")


def SingleRadarFileEstimationRainfall(file_path):
    radar = pyart.io.read_nexrad_archive(file_path)
    scan_time_minute = radar.time["data"][-1] / 60.0
    start, end = radar.get_start_end(sweep=0)
    assert end - start == 719

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

    print(minDifferenceIndex)
    print(azimuths[minDifferenceIndex])

    reflectivity = radar.fields["reflectivity"]["data"][start: end + 1, ...]
    differential_reflectivity = radar.fields["differential_reflectivity"]["data"][start: end + 1, ...]

    reflectivity_1 = reflectivity[minDifferenceIndex:, ...]
    reflectivity_2 = reflectivity[0:minDifferenceIndex, ...]
    reflectivity_north = numpy.ma.concatenate([reflectivity_1, reflectivity_2], axis=0)

    differential_reflectivity_1 = differential_reflectivity[minDifferenceIndex:, ...]
    differential_reflectivity_2 = differential_reflectivity[0:minDifferenceIndex, ...]
    differential_reflectivity_north = numpy.ma.concatenate([differential_reflectivity_1, differential_reflectivity_2],
                                                           axis=0)

    prediction = ma_QPEbyZhAndZdr(reflectivity_north, differential_reflectivity_north)
    # prediction = ma_QPEbyZh(reflectivity_north)
    rainfall = numpy.ma.filled(prediction, fill_value=0.0) / 60.0 * scan_time_minute
    return rainfall


def FolderRadarFileEstimationRainfall(folder_path: str, save_path: str):
    files = sorted(os.listdir(folder_path))
    estimation_list = []
    for filename in files:
        path = os.path.join(folder_path, filename)
        estimation_list.append(SingleRadarFileEstimationRainfall(path))

    estimation_array = numpy.stack(estimation_list, axis=0)
    print(estimation_array.shape)

    rainfall_sum = numpy.sum(estimation_array, axis=0)
    print(rainfall_sum.shape)

    first_radar = pyart.io.read_nexrad_archive(os.path.join(folder_path, files[0]))
    start, end = first_radar.get_start_end(sweep=0)
    assert end - start == 719

    azimuths = first_radar.azimuth["data"][start: end + 1]
    minDifferenceIndex = numpy.argmin(numpy.abs(azimuths - 0.0))
    azimuth1 = numpy.expand_dims(azimuths[minDifferenceIndex:], axis=0)
    azimuth2 = numpy.expand_dims(azimuths[:minDifferenceIndex], axis=0)
    azimuths = numpy.concatenate([azimuth1, azimuth2], axis=1)
    azimuths = numpy.squeeze(azimuths)

    elevations = first_radar.elevation["data"][start: end + 1]
    elevation1 = numpy.expand_dims(elevations[minDifferenceIndex:], axis=0)
    elevation2 = numpy.expand_dims(elevations[:minDifferenceIndex], axis=0)
    elevations = numpy.concatenate([elevation1, elevation2], axis=1)
    elevations = numpy.squeeze(elevations)
    print(minDifferenceIndex)
    print(azimuths[minDifferenceIndex])

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    EstimationPlotDomain(fig, ax, first_radar.range["data"], azimuths, elevations, rainfall_sum)
    plt.savefig(fname=save_path, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    FolderRadarFileEstimationRainfall("c:\\users\\admin\\desktop\\c", "tmp.png")
