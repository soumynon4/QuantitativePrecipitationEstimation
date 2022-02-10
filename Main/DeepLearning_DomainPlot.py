# @Time     : 1/10/2022 3:15 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : DeepLearning_DomainPlot.py
# @Project  : QuantitativePrecipitationEstimation
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyart
import torch

from Methods.ResNet import resnet101
from Kit.RadarGaugeDataset import normalization
from Kit.DomainPlotForEstimation import EstimationPlotDomain

matplotlib.use("Agg")


def findLowestDualSweepIndices(RadarClass: pyart.core.Radar, neededNumber: int):
    sweepNumbers = RadarClass.fixed_angle['data'].size
    Index_and_elevation = dict()
    __index = 0
    for sweep in range(sweepNumbers):
        sweepStart = RadarClass.sweep_start_ray_index['data'][sweep]
        sweepEnd = RadarClass.sweep_end_ray_index['data'][sweep] + 1
        thisSweepDifferentialReflectivity = RadarClass.fields['differential_reflectivity']['data'][sweepStart:sweepEnd]
        thisSweepDifferentialReflectivityMask = np.ma.getmask(thisSweepDifferentialReflectivity)
        isSingle = np.all(thisSweepDifferentialReflectivityMask)
        if isSingle:
            __index += 1
            continue
        else:
            Index_and_elevation.update({__index: RadarClass.fixed_angle['data'][sweep].item()})
            __index += 1
    sorted_index_and_elevation = sorted(Index_and_elevation.items(), key=lambda x: x[1])
    if len(sorted_index_and_elevation) < neededNumber:
        raise ValueError("radar only have {} dual-sweeps, but you need {} sweeps".format(
            len(sorted_index_and_elevation), neededNumber)
        )

    indices = [sorted_index_and_elevation[i][0] for i in range(neededNumber)]
    return indices


def sweepDataConcatenate(RadarClass: pyart.core.Radar, sweepIndex: int):
    start, end = RadarClass.get_start_end(sweep=sweepIndex)

    azimuths = RadarClass.azimuth["data"][start: end + 1]
    minDifferenceIndex = np.argmin(np.abs(azimuths - 0.0))

    azimuth1 = np.expand_dims(azimuths[minDifferenceIndex:], axis=0)
    azimuth2 = np.expand_dims(azimuths[:minDifferenceIndex], axis=0)
    azimuths = np.concatenate([azimuth1, azimuth2], axis=1)
    NorthAzimuths = np.squeeze(azimuths)

    elevations = RadarClass.elevation["data"][start: end + 1]
    elevation1 = np.expand_dims(elevations[minDifferenceIndex:], axis=0)
    elevation2 = np.expand_dims(elevations[:minDifferenceIndex], axis=0)
    elevations = np.concatenate([elevation1, elevation2], axis=1)
    NorthElevations = np.squeeze(elevations)

    reflectivity = RadarClass.fields["reflectivity"]["data"][start: end + 1, ...]
    differential_reflectivity = RadarClass.fields["differential_reflectivity"]["data"][start: end + 1, ...]

    reflectivity_1 = reflectivity[minDifferenceIndex:, ...]
    reflectivity_2 = reflectivity[0:minDifferenceIndex, ...]
    NorthReflectivity = np.ma.concatenate(
        [reflectivity_1, reflectivity_2],
        axis=0
    )
    differential_reflectivity_1 = differential_reflectivity[minDifferenceIndex:, ...]
    differential_reflectivity_2 = differential_reflectivity[0:minDifferenceIndex, ...]
    NorthDifferentialReflectivity = np.ma.concatenate(
        [differential_reflectivity_1, differential_reflectivity_2],
        axis=0
    )
    return NorthAzimuths, NorthElevations, NorthReflectivity, NorthDifferentialReflectivity


def SingleDualRadarDeepLearningDomainPlot(RadarFilePath: str, DeepLearningModel: torch.nn.Module, Device: torch.device):
    radar = pyart.io.read_nexrad_archive(RadarFilePath)
    scan_time_minute = radar.time["data"][-1] / 60.0
    lowestIndex = findLowestDualSweepIndices(radar, 2)

    for index_sweep in lowestIndex:
        start, end = radar.get_start_end(sweep=index_sweep)
        print("{} {}".format(start, end))
        assert end - start == 719

    _, _, ref0, zdr0 = sweepDataConcatenate(radar, lowestIndex[0])
    _, _, ref1, zdr1 = sweepDataConcatenate(radar, lowestIndex[1])
    mask = np.ma.getmask(ref0)

    REF = np.ma.stack([ref0, ref1], axis=0)
    ZDR = np.ma.stack([zdr0, zdr1], axis=0)
    REF_Range_Padding = np.ma.ones((2, 720, 4)) * -15.0
    ZDR_Range_Padding = np.ma.ones((2, 720, 4)) * -2.0

    REF = np.ma.concatenate([REF_Range_Padding, REF, REF_Range_Padding], axis=2)
    ZDR = np.ma.concatenate([ZDR_Range_Padding, ZDR, ZDR_Range_Padding], axis=2)

    REF_Azimuth_Padding1 = REF[:, -4:, :]
    REF_Azimuth_Padding2 = REF[:, :4, :]
    REF = np.ma.concatenate([REF_Azimuth_Padding1, REF, REF_Azimuth_Padding2], axis=1)

    ZDR_Azimuth_Padding1 = ZDR[:, -4:, :]
    ZDR_Azimuth_Padding2 = ZDR[:, :4, :]
    ZDR = np.ma.concatenate([ZDR_Azimuth_Padding1, ZDR, ZDR_Azimuth_Padding2], axis=1)

    REF = np.ma.filled(REF, fill_value=-15.0)
    ZDR = np.ma.filled(ZDR, fill_value=-2.0)

    # 预处理
    REF = normalization(REF, 0, 65, -0.5, 1.0, nan_value=0.0)
    ZDR = normalization(ZDR, -0.5, 3.5, -1.0, 1.0, nan_value=0.0)
    # end

    print(REF.shape)
    print(ZDR.shape)
    startRow = 4
    startCol = 4
    # endRow = 4 + 720 - 1
    # endCol = 4 + 1832 - 1

    dataset = []
    dataField = np.concatenate([REF, ZDR], axis=0)
    print(dataField.shape)

    for row in range(720):
        for column in range(1832):
            centerRow = startRow + row
            centerCol = startCol + column
            data = dataField[:, centerRow - 4: centerRow + 4 + 1, centerCol - 4: centerCol + 4 + 1]
            dataset.append(data)

    dataset = np.stack(dataset, axis=0)
    dataset = torch.from_numpy(dataset).float()
    print(dataset.size())

    DeepLearningModel.eval()
    with torch.no_grad():
        prediction = DeepLearningModel(dataset.to(Device))
    pred = prediction.cpu().numpy()
    pred = np.reshape(pred, (720, 1832))

    pred[mask] = 0.0
    pred[pred < 0.0] = 0.0

    pred = pred / 60.0 * scan_time_minute

    return pred


if __name__ == '__main__':
    runningDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet101(num_classes=1).to(device=runningDevice)
    model.load_state_dict(
        torch.load(
            "/media/data3/zhangjianchang/DATA/ModelCheckPoints/ResNet/ResNet-19.pth",
            map_location=runningDevice
        )
    )

    DualRadarFolder = "/media/data3/zhangjianchang/tmp"
    DualRadarFiles = sorted(os.listdir(DualRadarFolder))

    Origin_Radar = pyart.io.read_nexrad_archive(os.path.join(DualRadarFolder, DualRadarFiles[0]))
    Origin_Lowest_Index = findLowestDualSweepIndices(Origin_Radar, 2)

    for Origin_Index_Sweep in Origin_Lowest_Index:
        Origin_Start, Origin_End = Origin_Radar.get_start_end(sweep=Origin_Index_Sweep)
        print("{} {}".format(Origin_Start, Origin_End))
        assert Origin_End - Origin_Start == 719

    AzimuthFromNorth, ElevationFromNorth, _, _ = sweepDataConcatenate(Origin_Radar, Origin_Lowest_Index[0])
    RangeFromNorth = Origin_Radar.range["data"]

    DomainData = []

    for DualRadarFile in DualRadarFiles:
        estimation = SingleDualRadarDeepLearningDomainPlot(
            os.path.join(DualRadarFolder, DualRadarFile),
            model,
            runningDevice
        )
        DomainData.append(estimation)

    DomainData = np.sum(np.stack(DomainData, axis=0), axis=0)
    DomainData = np.squeeze(DomainData)

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    EstimationPlotDomain(fig, ax, RangeFromNorth, AzimuthFromNorth, ElevationFromNorth, DomainData)
    plt.savefig("/media/data3/zhangjianchang/tmp.png", bbox_inches="tight", pad_inches=0)
