# @Time     : 7/10/2021 9:34 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : DROPS.py
# @Project  : QuantitativePrecipitationEstimation
from collections import defaultdict

import numpy as np
import pyart
from netCDF4 import Dataset


def drops_reader(nc_path):
    radar = Dataset(nc_path)
    _scan_type = 'ppi'
    _time = change_time(radar)
    _range = change_range(radar)
    _metadata = change_metadata(radar)
    _latitude, _longitude, _altitude = change_geo(radar)
    _fixed_angle = change_fixed_angle(radar)
    _sweep_number = change_sweep_number(radar)
    _sweep_mode = change_sweep_mode(radar)
    _azimuth = change_azimuth(radar)
    _elevation = change_elevation(radar)
    _sweep_start_ray_index, _sweep_end_ray_index = change_sweep(radar)
    _fields = change_fields(radar)

    radar.close()

    return pyart.core.Radar(
        _time, _range, _fields, _metadata, _scan_type,
        _latitude, _longitude, _altitude,
        _sweep_number, _sweep_mode, _fixed_angle, _sweep_start_ray_index,
        _sweep_end_ray_index,
        _azimuth, _elevation)


def change_fields(radar_nc):
    length = radar_nc.variables["range"][:].shape[0]

    # 1.Change reflectivity field
    field_dict = defaultdict(dict)
    field_dict["reflectivity"]["units"] = 'dBZ'
    field_dict["reflectivity"]["standard_name"] = 'equivalent_reflectivity_factor'
    field_dict["reflectivity"]["long_name"] = 'Reflectivity'
    field_dict["reflectivity"]["valid_max"] = np.ma.max(radar_nc.variables["Reflectivity"][:])
    field_dict["reflectivity"]["valid_min"] = np.ma.min(radar_nc.variables["Reflectivity"][:])
    field_dict["reflectivity"]["coordinates"] = "elevation azimuth range"
    field_dict["reflectivity"]["_FillValue"] = np.ma.MaskedArray.get_fill_value(radar_nc.variables["Reflectivity"][:])
    field_dict["reflectivity"]["data"] = np.reshape(radar_nc.variables["Reflectivity"][:], (-1, length))

    # 2.Change Cross_Correlation_ratio
    field_dict["cross_correlation_ratio"]["units"] = 'ratio'
    field_dict["cross_correlation_ratio"]["standard_name"] = 'cross_correlation_ratio_hv'
    field_dict["cross_correlation_ratio"]["long_name"] = 'Cross correlation_ratio (RHOHV)'
    field_dict["cross_correlation_ratio"]["valid_max"] = 1.0
    field_dict["cross_correlation_ratio"]["valid_min"] = 0.0
    field_dict["cross_correlation_ratio"]["coordinates"] = "elevation azimuth range"
    field_dict["cross_correlation_ratio"]["_FillValue"] = np.ma.MaskedArray.get_fill_value(
        radar_nc.variables["CrossPolCorrelation"][:])
    field_dict["cross_correlation_ratio"]["data"] = np.reshape(radar_nc.variables["CrossPolCorrelation"][:],
                                                               (-1, length))

    # 3.Change Differential_Phase
    field_dict["differential_phase"]["units"] = 'degrees'
    field_dict["differential_phase"]["standard_name"] = 'differential_phase_hv'
    field_dict["differential_phase"]["long_name"] = 'differential_phase_hv'
    field_dict["differential_phase"]["valid_max"] = 360.0
    field_dict["differential_phase"]["valid_min"] = 0.0
    field_dict["differential_phase"]["coordinates"] = "elevation azimuth range"
    field_dict["differential_phase"]["_FillValue"] = np.ma.MaskedArray.get_fill_value(radar_nc.variables["PhiDP"][:])
    field_dict["differential_phase"]["data"] = np.reshape(radar_nc.variables["PhiDP"][:], (-1, length))

    # 4.Create KDP field
    field_dict["KDP"]["units"] = "degree/km"
    field_dict["KDP"]["standard_name"] = "Specific Differential Phase"
    field_dict["KDP"]["long_name"] = "Specific Differential Phase"
    field_dict["KDP"]["valid_max"] = np.ma.max(radar_nc.variables["KDP"][:])
    field_dict["KDP"]["valid_min"] = np.ma.min(radar_nc.variables["KDP"][:])
    field_dict["KDP"]["coordinates"] = 'elevation azimuth range'
    field_dict["KDP"]["_FillValue"] = np.ma.MaskedArray.get_fill_value(radar_nc.variables["KDP"][:])
    field_dict["KDP"]["data"] = np.reshape(radar_nc.variables["KDP"][:], (-1, length))

    # 5.Create ZDR field
    field_dict["ZDR"]["units"] = "dB"
    field_dict["ZDR"]["standard_name"] = "DifferentialReflectivity"
    field_dict["ZDR"]["long_name"] = "DifferentialReflectivity"
    field_dict["ZDR"]["valid_max"] = np.ma.max(radar_nc.variables["DifferentialReflectivity"][:])
    field_dict["ZDR"]["valid_min"] = np.ma.min(radar_nc.variables["DifferentialReflectivity"][:])
    field_dict["ZDR"]["coordinates"] = "elevation azimuth range"
    field_dict["ZDR"]["_FillValue"] = np.ma.MaskedArray.get_fill_value(
        radar_nc.variables["DifferentialReflectivity"][:])
    field_dict["ZDR"]["data"] = np.reshape(radar_nc.variables["DifferentialReflectivity"][:], (-1, length))

    return field_dict


def change_metadata(radar_nc):
    metadata_dict = {"Conventions": radar_nc.getncattr("Conventions"),
                     "version": radar_nc.getncattr("version"),
                     "title": '',
                     "institution": '',
                     "reference": '',
                     "source": '',
                     "history": '',
                     "comment": '',
                     "instrument_name": radar_nc.getncattr("instrument_name"),
                     "original_container": 'NEXRAD Level II',
                     "vcp_pattern": radar_nc.getncattr("scan_id")}
    return metadata_dict


def change_time(radar_nc):
    time_dict = {"units": radar_nc["time"].units,
                 "standard_name": "time",
                 "long_name": 'time_in_seconds_since_volume_start',
                 "calendar": 'gregorian',
                 # "comment": radar_nc.time["comment"],
                 "data": radar_nc["time"][:]}
    return time_dict


def change_range(radar_nc):
    range_dict = {"units": radar_nc["range"].units,
                  "standard_name": 'projection_range_coordinate',
                  "long_name": 'range_to_measurement_volume',
                  "axis": 'radial_range_coordinate',
                  "spacing_is_constant": str(radar_nc["range"].spacing_is_constant),
                  "comment": 'Coordinate variable for range. Range to center of each bin.',
                  "data": radar_nc["range"][:],
                  "meters_to_center_of_first_gate": radar_nc["range"].meters_to_center_of_first_gate,
                  "meters_between_gates": radar_nc["range"].meters_between_gates}
    return range_dict


def change_geo(radar_nc):
    latitude_dict = {"long_name": "Latitude",
                     "standard_name": "Latitude",
                     "units": "degrees_north",
                     "data": np.array([radar_nc["latitude"][:]])}
    longitude_dict = {"long_name": "Longitude",
                      "standard_name": "Longitude",
                      "units": "degrees_east",
                      "data": np.array([radar_nc["longitude"][:]])}
    altitude_dict = {"long_name": "Altitude",
                     "standard_name": "Altitude",
                     "units": "meters",
                     "positive": "up",
                     "data": np.array([radar_nc["altitude"][:]])}
    return latitude_dict, longitude_dict, altitude_dict


def change_sweep_number(radar_nc):
    sweep_number_dict = {"units": 'count',
                         "standard_name": 'sweep_number',
                         "long_name": 'Sweep number',
                         "data": radar_nc["sweep_number"][:]}
    return sweep_number_dict


def change_sweep_mode(radar_nc):
    sweep_mode_dict = {"units": "unitless",
                       "standard_name": "sweep_mode",
                       "long_name": "Sweep mode",
                       # "comment": radar_nc.sweep_mode["comment"],
                       "data": radar_nc["sweep_mode"][:]}
    return sweep_mode_dict


def change_fixed_angle(radar_nc):
    fixed_angle_dict = {"long_name": 'Target angle for sweep',
                        "units": "degrees",
                        "standard_name": 'target_fixed_angle',
                        "data": np.array([radar_nc.variables["elevation"][:][i] for i in radar_nc.variables["sweep_start_ray_index"][:]])}
    return fixed_angle_dict


def change_sweep(radar_nc):
    sweep_start_ray_index_dict = {"units": "count",
                                  "long_name": 'Index of first ray in sweep, 0-based',
                                  "data": radar_nc["sweep_start_ray_index"][:]}

    sweep_end_ray_index_dict = {"unit": "count",
                                "long_name": 'Index of last ray in sweep, 0-based',
                                "data": radar_nc["sweep_end_ray_index"][:]}
    return sweep_start_ray_index_dict, sweep_end_ray_index_dict


def change_azimuth(radar_nc):
    azimuth_dict = {"units": "degrees",
                    "standard_name": 'beam_azimuth_angle',
                    "long_name": 'azimuth_angle_from_true_north',
                    "axis": 'radial_azimuth_coordinate',
                    "comment": 'Azimuth of antenna relative to true north',
                    "data": radar_nc["azimuth"][:]}
    return azimuth_dict


def change_elevation(radar_nc):
    elevation_dict = {"units": "degrees",
                      "standard_name": "beam_elevation_angle",
                      "long_name": 'elevation_angle_from_horizontal_plane',
                      "axis": 'radial_elevation_coordinate',
                      "comment": 'Elevation of antenna relative to the horizontal plane',
                      "data": radar_nc["elevation"][:]}
    return elevation_dict
