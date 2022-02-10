# @Time     : 7/10/2021 9:12 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : Radar.py
# @Project  : QuantitativePrecipitationEstimation
import collections
import datetime
from pprint import pprint
from typing import Tuple

import numpy as np
import pyart


def determine_time_for_radar(RadarFileName: str) -> datetime.datetime:
    radar_time = datetime.datetime(year=int(RadarFileName[4:8]), month=int(RadarFileName[8:10]),
                                   day=int(RadarFileName[10:12]), hour=int(RadarFileName[13:15]),
                                   minute=int(RadarFileName[15:17]), second=int(RadarFileName[17:19]))
    return radar_time


def radar_time_gauge_time(radar_file: str, scan_seconds: float) -> Tuple[datetime.datetime, datetime.datetime]:
    radarTime_start = determine_time_for_radar(radar_file)
    radarTime_end = radarTime_start + datetime.timedelta(seconds=int(scan_seconds))
    print("Radar Start: {}, End: {}".format(radarTime_start, radarTime_end))

    StartTime = radarTime_start - datetime.timedelta(seconds=radarTime_start.second) + datetime.timedelta(minutes=1)

    # if radarTime_start.second <= 30: StartTime = radarTime_start - datetime.timedelta(
    # seconds=radarTime_start.second) + datetime.timedelta(minutes=1) else: StartTime = radarTime_start -
    # datetime.timedelta(seconds=radarTime_start.second) + datetime.timedelta(minutes=2)

    if radarTime_end.second >= 50:
        EndTime = radarTime_end - datetime.timedelta(seconds=radarTime_end.second) + datetime.timedelta(minutes=1)
    else:
        EndTime = radarTime_end - datetime.timedelta(seconds=radarTime_end.second)

    return StartTime, EndTime


def getRadarData(radar: pyart.core.Radar, start_time: datetime.datetime, target_azimuth: float, target_range: float,
                 window_size: tuple = (9, 9), max_elevation: float = 3.0, numbers_elevation=2,
                 reflectivity_filled_value: float = np.nan, differential_reflectivity_filled_value: float = np.nan):
    if not isinstance(target_azimuth, float):
        raise TypeError('{} is not Float Class'.format(target_azimuth))
    if not isinstance(target_range, float):
        raise TypeError('{} is not Float Class'.format(target_range))
    if window_size[0] & 1 != 1:
        raise ValueError('{} is even.'.format(window_size[0]))
    elif window_size[1] & 1 != 1:
        raise ValueError('{} is even.'.format(window_size[1]))

    targetAzimuth = target_azimuth
    targetRange = target_range  # unit: meter
    azimuthWindowSize = window_size[0] // 2
    rangeWindowSize = window_size[1] // 2

    sweepNumbers = radar.fixed_angle['data'].size
    Index_and_elevation = dict()
    __index = 0
    fields = collections.defaultdict(list)
    data = collections.defaultdict(list)

    final_data = dict()
    final_fields = dict()

    for sweep in range(sweepNumbers):
        sweepElevation = radar.fixed_angle['data'][sweep].item()
        if sweepElevation > max_elevation:
            continue

        sweepStart = radar.sweep_start_ray_index['data'][sweep]
        sweepEnd = radar.sweep_end_ray_index['data'][sweep] + 1

        thisSweepDifferentialReflectivity = radar.fields['differential_reflectivity']['data'][sweepStart:sweepEnd]

        thisSweepDifferentialReflectivityMask = np.ma.getmask(thisSweepDifferentialReflectivity)
        isSingle = np.all(thisSweepDifferentialReflectivityMask)
        if isSingle:
            continue
        else:
            print("This *{}* sweep is Dual-polar".format(sweep))

        thisSweepAzimuth = radar.azimuth['data'][sweepStart:sweepEnd]
        thisSweepRange = radar.range['data']

        thisSweepTargetAzimuthIndex = np.argmin(np.abs(thisSweepAzimuth - targetAzimuth))
        thisSweepTargetRangeIndex = np.argmin(np.abs(thisSweepRange - targetRange))

        targetAzimuthIndex = sweepStart + thisSweepTargetAzimuthIndex
        targetRangeIndex = thisSweepTargetRangeIndex

        ray_time = start_time + datetime.timedelta(seconds=int(radar.time['data'][targetAzimuthIndex].item()))
        Index_and_elevation.update({__index: radar.fixed_angle['data'][sweep].item()})
        print("Sweep Index: {}".format(__index))
        __index += 1

        print("Target azimuth index: {} in this layer.".format(thisSweepTargetAzimuthIndex))
        print("Target range index: {} in this layer.".format(thisSweepTargetRangeIndex))

        print("This sweep first elevation {}".format(radar.elevation['data'][sweepStart]))
        print("This ray elevation {}".format(radar.elevation["data"][targetAzimuthIndex]))
        print("This sweep fixed angle {}".format(sweepElevation))
        print("This ray time: {}".format(ray_time))

        print("-- Verification Azimuth Start --")
        print("Target Azimuth: {}.".format(targetAzimuth))
        print("Azimuth: {} in All Sweeps.".format(radar.azimuth["data"][targetAzimuthIndex].item()))
        print("Azimuth: {} in this Sweep.".format(thisSweepAzimuth[thisSweepTargetAzimuthIndex]))

        if radar.azimuth['data'][targetAzimuthIndex].item() != thisSweepAzimuth[thisSweepTargetAzimuthIndex]:
            raise ValueError('-- Azimuth Calculate Wrong. --')
        print("-- Verification Azimuth End --")

        print("-- Verification Range Start --")
        print("Target Range : {} meters".format(targetRange))
        print("Calculate Range: {} meters".format(radar.range["data"][targetRangeIndex].item()))
        print("-- Verification Range End --")

        reflectivity = np.ma.filled(
            radar.fields['reflectivity']['data'][targetAzimuthIndex, targetRangeIndex],
            fill_value=reflectivity_filled_value
        ).item()  # float

        differential_reflectivity = np.ma.filled(
            radar.fields['differential_reflectivity']['data'][targetAzimuthIndex, targetRangeIndex],
            fill_value=differential_reflectivity_filled_value
        ).item()  # float

        if thisSweepTargetAzimuthIndex - azimuthWindowSize >= 0 and (
                sweepEnd - 1 - targetAzimuthIndex - azimuthWindowSize) >= 0:
            print("Field Method 1")

            reflectivity_field = np.ma.filled(
                radar.fields['reflectivity']['data'][targetAzimuthIndex - azimuthWindowSize:
                                                     targetAzimuthIndex + azimuthWindowSize + 1,
                targetRangeIndex - rangeWindowSize:
                targetRangeIndex + rangeWindowSize + 1],
                fill_value=reflectivity_filled_value
            )  # array

            differential_reflectivity_field = np.ma.filled(
                radar.fields['differential_reflectivity']['data'][targetAzimuthIndex - azimuthWindowSize:
                                                                  targetAzimuthIndex + azimuthWindowSize + 1,
                targetRangeIndex - rangeWindowSize:
                targetRangeIndex + rangeWindowSize + 1],
                fill_value=differential_reflectivity_filled_value
            )

            print('--- Validate Filed Azimuth Start ---')
            print("Sweep start: {}, Sweep end: {}, target index: {}".format(sweepStart, sweepEnd, targetAzimuthIndex))
            print("Field index start: {}, end: {}".format(
                targetAzimuthIndex - azimuthWindowSize,
                targetAzimuthIndex + azimuthWindowSize + 1)
            )
            print(
                radar.azimuth['data'][targetAzimuthIndex - azimuthWindowSize:
                                      targetAzimuthIndex + azimuthWindowSize + 1]
            )

            if radar.azimuth['data'][targetAzimuthIndex - azimuthWindowSize: targetAzimuthIndex + azimuthWindowSize + 1].size != window_size[0]:
                raise ValueError('!!!Mismatch!!!')
            print('--- Validate Filed Azimuth End ---')

        else:
            if thisSweepTargetAzimuthIndex - azimuthWindowSize < 0:
                print("Field Method 2")
                difference = thisSweepTargetAzimuthIndex - azimuthWindowSize
                reflectivity_field_1 = np.ma.filled(
                    radar.fields['reflectivity']['data'][sweepEnd + difference:
                                                         sweepEnd,
                                                         targetRangeIndex - rangeWindowSize:
                                                         targetRangeIndex + rangeWindowSize + 1],
                    fill_value=reflectivity_filled_value
                )

                reflectivity_field_2 = np.ma.filled(
                    radar.fields['reflectivity']['data'][sweepStart:
                                                         targetAzimuthIndex + azimuthWindowSize + 1,
                                                         targetRangeIndex - rangeWindowSize:
                                                         targetRangeIndex + rangeWindowSize + 1],
                    fill_value=reflectivity_filled_value
                )

                print('--- Validate Filed Azimuth Start ---')
                print("Sweep start: {}, Sweep end: {}, target index: {}".format(
                    sweepStart, sweepEnd, targetAzimuthIndex)
                )
                print("Field index start: {}, end: {}".format(
                    sweepEnd + difference, targetAzimuthIndex + azimuthWindowSize + 1)
                )
                print(radar.azimuth['data'][sweepEnd + difference:sweepEnd])
                print(radar.azimuth['data'][sweepStart:targetAzimuthIndex + azimuthWindowSize + 1])

                if radar.azimuth['data'][sweepEnd + difference:sweepEnd].size + radar.azimuth['data'][sweepStart:targetAzimuthIndex + azimuthWindowSize + 1].size != window_size[0]:
                    raise ValueError('!!!Mismatch!!!')
                print('--- Validate Filed Azimuth End ---')

                differential_reflectivity_field_1 = np.ma.filled(radar.fields['differential_reflectivity']['data'][
                                                                 sweepEnd + difference:
                                                                 sweepEnd,
                                                                 targetRangeIndex - rangeWindowSize:
                                                                 targetRangeIndex + rangeWindowSize + 1],
                                                                 fill_value=differential_reflectivity_filled_value)

                differential_reflectivity_field_2 = np.ma.filled(radar.fields['differential_reflectivity']['data'][
                                                                 sweepStart:
                                                                 targetAzimuthIndex + azimuthWindowSize + 1,
                                                                 targetRangeIndex - rangeWindowSize:
                                                                 targetRangeIndex + rangeWindowSize + 1],
                                                                 fill_value=differential_reflectivity_filled_value)

                reflectivity_field = np.concatenate(
                    (reflectivity_field_1, reflectivity_field_2),
                    axis=0
                )

                differential_reflectivity_field = np.concatenate(
                    (differential_reflectivity_field_1, differential_reflectivity_field_2),
                    axis=0
                )

            elif (sweepEnd - 1 - targetAzimuthIndex - azimuthWindowSize) < 0:
                print("Field Method 3")
                difference = sweepEnd - 1 - targetAzimuthIndex - azimuthWindowSize
                reflectivity_field_1 = np.ma.filled(
                    radar.fields['reflectivity']['data'][targetAzimuthIndex - azimuthWindowSize:
                                                         sweepEnd,
                                                         targetRangeIndex - rangeWindowSize:
                                                         targetRangeIndex + rangeWindowSize + 1],
                    fill_value=reflectivity_filled_value
                )

                reflectivity_field_2 = np.ma.filled(
                    radar.fields['reflectivity']['data'][sweepStart:
                                                         sweepStart - difference,
                                                         targetRangeIndex - rangeWindowSize:
                                                         targetRangeIndex + rangeWindowSize + 1],
                    fill_value=reflectivity_filled_value
                )

                print('--- Validate Filed Azimuth Start ---')
                print("Sweep start: {}, Sweep end: {}, target index: {}".format(
                    sweepStart, sweepEnd, targetAzimuthIndex)
                )
                print("Field index start: {}, end: {}".format(
                    targetAzimuthIndex - azimuthWindowSize, sweepStart - difference)
                )
                print(radar.azimuth['data'][targetAzimuthIndex - azimuthWindowSize:sweepEnd])
                print(radar.azimuth['data'][sweepStart:sweepStart - difference])

                if radar.azimuth['data'][targetAzimuthIndex - azimuthWindowSize:sweepEnd].size + radar.azimuth['data'][sweepStart:sweepStart - difference].size != window_size[0]:
                    raise ValueError('!!!Mismatch!!!')
                print('--- Validate Filed Azimuth End ---')

                differential_reflectivity_field_1 = np.ma.filled(
                    radar.fields['differential_reflectivity']['data'][targetAzimuthIndex - azimuthWindowSize:
                                                                      sweepEnd,
                                                                      targetRangeIndex - rangeWindowSize:
                                                                      targetRangeIndex + rangeWindowSize + 1],
                    fill_value=differential_reflectivity_filled_value
                )

                differential_reflectivity_field_2 = np.ma.filled(
                    radar.fields['differential_reflectivity']['data'][sweepStart:
                                                                      sweepStart - difference,
                                                                      targetRangeIndex - rangeWindowSize:
                                                                      targetRangeIndex + rangeWindowSize + 1],
                    fill_value=differential_reflectivity_filled_value
                )

                reflectivity_field = np.concatenate(
                    (reflectivity_field_1, reflectivity_field_2),
                    axis=0
                )

                differential_reflectivity_field = np.concatenate(
                    (differential_reflectivity_field_1, differential_reflectivity_field_2),
                    axis=0
                )

            else:
                raise ValueError('!!!!!!Window Size Maybe Mismatch!!!!!!')

        if np.all(np.isnan(reflectivity_field)):
            print("{} reflectivity field of this layer is all Nan.".format(reflectivity_field.shape))
        else:
            print('The max reflectivity Value in {} field of this layer: {}'.format(
                reflectivity_field.shape,
                np.nanmax(reflectivity_field)
            ))

        if np.all(np.isnan(differential_reflectivity_field)):
            print("{} differential reflectivity field of this layer is all Nan.".format(
                differential_reflectivity_field.shape
            ))
        else:
            print('The max differential_reflectivity Value in {} filed of this layer: {}'.format(
                differential_reflectivity_field.shape,
                np.nanmax(differential_reflectivity_field)
            ))

        print('This location\'s reflectivity: {}'.format(reflectivity))
        print('This location\'s differential_reflectivity:{}'.format(differential_reflectivity))

        data['reflectivity'].append(reflectivity)
        data['differential_reflectivity'].append(differential_reflectivity)

        fields['reflectivity'].append(reflectivity_field)
        fields['differential_reflectivity'].append(differential_reflectivity_field)

        print("This Sweep reflectivity field:")
        pprint(reflectivity_field)
        print("This Sweep differential reflectivity field:")
        pprint(differential_reflectivity_field)

        # Split Line
        print('=' * 30)

    print("Index and elevation:")
    print(Index_and_elevation)

    sorted_index_and_elevation = sorted(Index_and_elevation.items(), key=lambda x: x[1])

    if len(sorted_index_and_elevation) < numbers_elevation:
        print("{} count of needed sweep is not enough.".format(len(sorted_index_and_elevation)))
        return final_data, final_fields

    indices = [sorted_index_and_elevation[i][0] for i in range(numbers_elevation)]

    print("sorted index and elevation:")
    pprint(sorted_index_and_elevation)
    print("Indices: {}".format(indices))

    final_data["reflectivity"] = np.array([data["reflectivity"][j] for j in indices])
    final_data["differential_reflectivity"] = np.array([data["differential_reflectivity"][j] for j in indices])

    print("The 1st - {}th lowest elevation {}".format(numbers_elevation, [Index_and_elevation[i] for i in indices]))
    print("The 1st - {}th lowest elevation's reflectivity: {}".format(numbers_elevation, final_data["reflectivity"]))
    print("The 1st - {}th lowest elevation's differential reflectivity: {}".format(numbers_elevation, final_data[
        "differential_reflectivity"]))

    final_fields["reflectivity"] = np.stack(
        [fields["reflectivity"][j] for j in indices],
        axis=0
    )

    final_fields["differential_reflectivity"] = np.stack(
        [fields["differential_reflectivity"][j] for j in indices],
        axis=0
    )
    for i in range(numbers_elevation):
        if np.all(np.isnan(final_fields["reflectivity"][i, ...])) or np.all(np.isnan(final_fields["differential_reflectivity"][i, ...])):
            print("{} Sweep Data all nan, drop this sample.".format(i))
            print("Reflectivity Field:")
            pprint(final_fields["reflectivity"])
            print("Differential Reflectivity Field:")
            pprint(final_fields["differential_reflectivity"])
            empty_data = dict()
            empty_fields = dict()
            return empty_data, empty_fields

    print("Reflectivity Field:")
    pprint(final_fields["reflectivity"])
    print("Differential Reflectivity Field:")
    pprint(final_fields["differential_reflectivity"])
    return final_data, final_fields
