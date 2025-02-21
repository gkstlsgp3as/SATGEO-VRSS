# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:39:52 2024

@author: Soyeon Park

Description: Make ECMWF forecast results ready to use. The code is applied to every subfolders in --folder.

How to use: python ECMWF_API.py --save <savepath> --folder <saved grib path from ECMWF_API.py> --geojson <geojson file with abs path> --maxhour <maximum forecast hour>
Example usage: python D:\Oilspill_Total\Code_Python\Wave_spectra\KCGSA_Codes\2_Ready2use_ECMWF.py 
                -s D:\Extract_spectra\ECMWF_forecast_ready -g "D:/TROPOMI/API/map.geojson" -f D:\Extract_spectra\ECMWF_forecast -m 48

"""

import os
import re
import xarray as xr
import numpy as np
import pandas as pd
from typing import List, Tuple
import json
from shapely.geometry import shape
from sqlalchemy.orm import Session

from app.config.db_connection import get_db
from app.config.settings import settings
from decimal import Decimal


# Extract date and time information from the directory path and determine file patterns.
#
#     Args:
#     - dir (str): Directory path.
#
#     Returns:
#     - Tuple containing wind pattern, wave pattern, date part, and time part.
from app.models.ecmwf_prediction_hist import EcmwfPredictionHist
from app.service.ecmwf_prediction_hist_service import bulk_upsert_ecmwf_prediction_hist, \
    bulk_insert_ecmwf_prediction_history


def parse_dir_info(dir: str) -> Tuple[str, str, str, str]:
    time_part = os.path.basename(dir)
    date_part = os.path.basename(os.path.dirname(dir))
    oper_pat = r"-oper-" if time_part in ['00z', '12z'] else r"-scda-"
    wave_pat = r"-wave-" if time_part in ['00z', '12z'] else r"-scwv-"
    return oper_pat, wave_pat, date_part, time_part


# Filter GRIB2 files and load them as xarray datasets.
#
#     Args:
#     - dir (str): Directory path.
#     - pat (str): File pattern to filter.
#     - max_hr (int): Maximum forecast hour.
#
#     Returns:
#     - List of xarray datasets and list of forecast hours.
def filter_and_load_grib(dir: str, pat: str, max_hr: int) -> Tuple[List[xr.Dataset], List[int]]:
    files = [f for f in os.listdir(dir) if f.endswith('.grib2') and pat in f]
    filtered = [f for f in files if int(re.search(r"-(\d+)h-", f).group(1)) <= max_hr]
    filtered.sort(key=lambda x: int(re.search(r"-(\d+)h-", x).group(1)))

    ds_list = []
    hours = []
    for file in filtered:
        path = os.path.join(dir, file)
        ds = xr.open_dataset(path, engine='cfgrib',
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}) \
            if 'oper' in pat or 'scda' in pat else xr.open_dataset(path, engine='cfgrib')
        ds_list.append(ds)
        hours.append(int(re.search(r"-(\d+)h-", file).group(1)))
    return ds_list, hours


# Combine the time and trim by latitude and longitude bounds from a GeoJSON file.
#
#     Args:
#     - ds_list (List[xr.Dataset]): List of xarray datasets.
#     - hours (List[int]): List of forecast hours.
#     - geojson (str): Path to the GeoJSON file containing the region of interest.
#
#     Returns:
#     - Trimmed xarray dataset.
def combine_and_trim(ds_list: List[xr.Dataset], hours: List[int], geojson: str) -> xr.Dataset:
    with open(geojson) as f:
        geojson_data = json.load(f)
    polygon = shape(geojson_data['features'][0]['geometry'])
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    times = pd.to_datetime(hours, unit='h', origin=pd.Timestamp(ds_list[0].time.values))
    ds_combined = xr.concat(ds_list, pd.Index(times, name='time'))
    ds_trimmed = ds_combined.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))

    return ds_trimmed


# Process forecast files by filtering, loading, combining, and trimming datasets.
#
#     Args:
#     - dir (str): Directory path.
#     - pat (str): File pattern to filter.
#     - max_hr (int): Maximum forecast hour.
#     - geojson (str): Path to the GeoJSON file containing the region of interest.
#
#     Returns:
#     - Trimmed xarray dataset and forecast datetime index.
def process_files(dir: str, pat: str, max_hr: int, geojson: str) -> Tuple[xr.Dataset, pd.DatetimeIndex]:
    ds_list, hours = filter_and_load_grib(dir, pat, max_hr)
    ds_trimmed = combine_and_trim(ds_list, hours, geojson)
    return ds_trimmed, pd.to_datetime(hours, unit='h', origin=ds_trimmed.time.values[0])


# Interpolate dataset to hourly intervals.
#
#     Args:
#     - ds_trimmed (xr.Dataset): Trimmed dataset.
#     - periods (int): Number of periods to interpolate.
#
#     Returns:
#     - Interpolated dataset.
def interpolate_data(ds_trimmed: xr.Dataset, periods: int) -> xr.Dataset:
    new_times = pd.date_range(start=ds_trimmed.time.values[0], periods=periods, freq='h')
    return ds_trimmed.interp(time=new_times, method='linear')


# Main function to process and save forecast data.
#
#     Args:
#     - input_dir (str): Path to the folder containing forecast data.
#     - output_dir (str): Path to save processed data.
#     - geojson (str): Path to the GeoJSON file containing the region of interest.
#     - max_hr (int): Maximum forecast hour.
def ready_to_use_ecmwf(db: Session, max_hr: int):
    input_dir = settings.W01_OUTPUT_PATH
    geojson = settings.W03_RESOURCE_PATH + "/map.geojson"

    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for base in subfolders:
        print(f"Processing {base}...")
        time_dirs = [os.path.join(base, t) for t in ['00z', '06z', '12z', '18z']]

        for dir in time_dirs:
            try:
                oper_pat, wave_pat, date_part, time_part = parse_dir_info(dir)

                # Process wind data
                ds_oper_trimmed, times = process_files(dir, oper_pat, max_hr, geojson)
                ds_oper_interp = interpolate_data(ds_oper_trimmed, max_hr)

                # Process wave data
                ds_wave_trimmed, _ = process_files(dir, wave_pat, max_hr, geojson)
                ds_wave_interp = interpolate_data(ds_wave_trimmed, max_hr)

                # Calculate wind speed and direction
                u, v = ds_oper_interp['u10'], ds_oper_interp['v10']
                speed = np.sqrt(u ** 2 + v ** 2)
                direction = np.mod(180 + np.arctan2(u, v) * (180 / np.pi), 360)

                # Create the dataset
                processed_data = xr.Dataset({
                    'wind_speed_real': speed,
                    'wind_direction_real': direction,
                    'wave_direction_real': ds_wave_interp['mwd'],
                    'wave_height_real': ds_wave_interp['swh'],
                })

                # Save as dataframe
                processed_dataframe = processed_data.to_dataframe().reset_index()
                processed_dataframe['prediction_standard_time'] = processed_dataframe['time']
                processed_dataframe['prediction_time'] = processed_dataframe['time']
                processed_dataframe['longitude_length'] = 0.25
                processed_dataframe['latitude_length'] = 0.25
                processed_dataframe['relative_humidity'] = 0.0
                processed_dataframe['total_column_water_vapor'] = 0.0
                processed_dataframe['skin_temperature'] = 0.0

                # 사용하지 않는 열 제거
                columns_to_drop = ["time", "heightAboveGround", "surface"]
                processed_dataframe = processed_dataframe.drop(columns=columns_to_drop)

                bulk_upsert_ecmwf_prediction_hist(db, processed_dataframe)

                # # Save the dataset
                # if 'latitude' in processed_data.coords and 'longitude' in processed_data.coords:
                #     processed_data = processed_data.rename({'latitude': 'lat', 'longitude': 'lon'})
                # filename = f"{date_part}_{time_part}.nc"
                # processed_data.to_netcdf(os.path.join(output_dir, filename))

                # # Save to db
                # bulk_data = []
                # time_size = processed_data['time'].size
                # lat_size = processed_data['latitude'].size
                # lon_size = processed_data['longitude'].size
                #
                # for i in range(time_size):
                #     for j in range(lat_size):
                #         for k in range(lon_size):
                #             try:
                #                 record = EcmwfPredictionHist(
                #                     prediction_standard_time=times[i],
                #                     prediction_time_timestamp=times[i],
                #                     latitude=Decimal(processed_data['latitude'][j].item()),
                #                     longitude=Decimal(processed_data['longitude'][k].item()),
                #                     latitude_length=float(processed_data['latitude'][j].item()),
                #                     longitude_length=float(processed_data['longitude'][k].item()),
                #                     wind_speed_real=float(processed_data['speed'][i, j, k].item()),
                #                     wind_direction_real=float(processed_data['dir'][i, j, k].item()),
                #                     wave_height_real=float(processed_data['waveh'][i, j, k].item()),
                #                     wave_direction_real=float(processed_data['waved'][i, j, k].item()),
                #                     relative_humidity=0.0,  # 필요한 값으로 교체
                #                     total_column_water_vapor=0.0,  # 필요한 값으로 교체
                #                     skin_temperature=0.0  # 필요한 값으로 교체
                #                 )
                #                 bulk_data.append(record)
                #
                #             except IndexError as index_error:
                #                 print(f"IndexError time:{i} lat:{j} lon:{k}. {index_error}")
                #                 continue
                #
                # # Bulk insert
                # result = bulk_insert_ecmwf_prediction_history(db, bulk_data)
                # print(f"bulk_data size: {result}")

            except Exception as e:
                print(f"Failed to process {dir}: {e}")
                continue


# if __name__ == "__main__":
#     db = next(get_db())
#     ready_to_use_ecmwf(db, 48)
