import argparse
import glob
import os
from datetime import datetime
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.service import noaa_prediction_hist_service

from app.config.db_connection import get_db

local_file_path = settings.W01_NOAA_OUTPUT_PATH + "/"
resource_file_path = settings.W03_NOAA_RESOURCE_PATH + "/"
batch_size = 1000
max_predication_hour = 48  # default
data_interval_hour = 3  # default

def parse_dir_info(target_datetime: str, sub_path: str, filename_pattern: str, num_index: int, split_delimiter: str) -> \
        List[str]:

    target_date = target_datetime[0:8]
    full_path = os.path.join(local_file_path, target_date, sub_path)
    all_files = glob.glob(os.path.join(full_path, filename_pattern))

    filtered_files = []
    target_time = int(target_datetime[8:10])

    # 추후 temporal interpolate에서 외삽을 피하고 보간으로만 진행하기 위해 더 이전의 데이터도 포함하기 위해 파일을 넉넉하게 가져감
    adjust_min_target_time = adjust_min_time(target_time)
    adjust_max_target_time = max_predication_hour + target_time + data_interval_hour
    for file in all_files:
        base_name = os.path.basename(file)
        number_str = base_name.split(split_delimiter)[num_index][1:4]  # 파일명에서 특정 인덱스 추출
        number = int(number_str)

        if adjust_min_target_time <= number <= adjust_max_target_time:
            filtered_files.append(file)

    filtered_files.sort()
    return filtered_files


def adjust_min_time(time_str: str) -> int:
    time_int = int(time_str)
    return time_int - (time_int % data_interval_hour)


def filter_coord(df: pd.DataFrame, buffer: bool = False) -> pd.DataFrame:
    lon_min, lon_max = 120, 140
    lat_min, lat_max = 25, 45
    if buffer:
        lon_min -= 0.25
        lon_max += 0.25
        lat_min -= 0.25
        lat_max += 0.25
    return df[(df['lon'] >= lon_min) & (df['lon'] <= lon_max) & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)]


def process_files(file_path_list: List[str], column_list: List[str], target_datetime: str) -> xr.Dataset:
    result_ds = []
    times_list = []

    for index, file_path in enumerate(file_path_list):
        # read csv
        df = pd.read_csv(file_path)

        # range trim
        filtered_df = filter_coord(df)

        valid_date_value = filtered_df['date'].dropna().values[0]

        # interpolation (spatial) - 그리드에서의 보간 수행
        for value in column_list:
            print("spatial interpolate start:" + value)
            if filtered_df[value].isna().any():
                interpolate_df = spatial_interpolate_griddata(filtered_df, value)
                # filtered_df[value] = interpolate_df.set_index(filtered_df.index)[value]
                filtered_df.loc[:, value] = interpolate_df.set_index(filtered_df.index)[value]
            else:
                print("nothing spatial interpolated")

        # Convert DataFrame to Dataset
        filtered_ds = filtered_df.set_index(['lat', 'lon']).to_xarray()

        result_ds.append(filtered_ds)
        times_list.append(np.datetime64(pd.to_datetime(valid_date_value)))

    # interpolation (temporal)
    times_list.sort()
    ds_combined = xr.concat(result_ds, pd.Index(times_list, name='time'))
    start_time = str_to_npdatetime(target_datetime)
    return temporal_interpolate_data(ds_combined, start_time)


def str_to_npdatetime(target_datetime_str: str) -> np.datetime64:
    target_datetime = datetime.strptime(target_datetime_str, "%Y%m%d%H%M%S")
    return np.datetime64(target_datetime)


def spatial_interpolate_griddata(filter_df: pd.DataFrame, value: str) -> pd.DataFrame:
    # 복사본 생성해서 비교용으로 씀
    filter_df = filter_coord(filter_df, True)
    old_filter_df = filter_df.copy()

    # NaN이 있는 부분만 보간할 것이므로 NaN 마스크 생성
    nan_mask = filter_df[value].isna()

    # 기존 데이터의 좌표 (lat, lon)와 대응하는 값
    points = np.array(filter_df[['lat', 'lon']])
    values = filter_df[value].values

    # NaN이 아닌 값들을 추출 (보간용)
    known_points = points[~nan_mask]
    known_values = values[~nan_mask]

    # NaN인 좌표만 추출 (보간 대상)
    nan_points = points[nan_mask]

    # NaN 값이 있는 위치에 대해 보간 수행 (linear 방법)
    interpolated_values = griddata(known_points, known_values, nan_points, method='linear')

    # 보간된 값을 원래 NaN 위치에 채워넣음
    values[nan_mask] = interpolated_values

    # NaN 값만 채워진 filter_df 반환
    filter_df[value] = values

    # 차트로 보간내용 확인 -- 테스트용
    # print_compare_chart(old_filter_df, filter_df, value)

    selected_columns = ['lat', 'lon', value]
    selected_df = filter_df[selected_columns]
    selected_df = selected_df.reset_index()

    land_point_df = extract_land_points()

    # selected_df에서 각각의 lat, lon을 순회
    for i in range(len(selected_df)):
        lat, lon = selected_df.loc[i, 'lat'], selected_df.loc[i, 'lon']

        # geo_df에서 같은 lat, lon이 있는지 확인
        if ((land_point_df['lat'] == lat) & (land_point_df['lon'] == lon)).any():
            # 동일한 lat, lon이 있으면 value를 NaN으로 설정
            selected_df.loc[i, value] = np.nan

    # print_compare_chart(old_filter_df, selected_df, value)

    result_df = filter_coord(selected_df)

    return result_df


def extract_land_points() -> pd.DataFrame:
    # Natural Earth Landmass Shapefile 경로
    landmass_file_path = resource_file_path + 'ne_10m_land.shx'

    # Shapefile 읽기
    world = gpd.read_file(landmass_file_path)

    # 육지(다각형) 정보만 선택 (해양 정보 제외)
    land = world[world['geometry'].type == 'MultiPolygon']

    # 0.25도 간격으로 위도와 경도 생성
    lon_min, lon_max = 120, 140
    lat_min, lat_max = 25, 45
    lat_range = np.arange(lat_min, lat_max + 0.25, 0.25)
    lon_range = np.arange(lon_min, lon_max + 0.25, 0.25)

    # 그리드 생성
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    # 좌표를 DataFrame으로 변환
    grid_df = pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten()
    })

    # 그리드의 좌표를 Point 객체로 변환
    grid_points = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat))

    # 육지와 겹치는 좌표만 필터링
    land_points = gpd.sjoin(grid_points, land, how="inner", predicate="within")

    # 육지 좌표 출력
    # print(land_points[['lat', 'lon']])

    return land_points


def temporal_interpolate_data(ds_trimmed: xr.Dataset, start_time: np.datetime64) -> xr.Dataset:
    # 3h간격이던 데이터를 1h간격으로 linear interpolate
    new_times = pd.date_range(start=start_time, periods=max_predication_hour, freq='h')
    return ds_trimmed.interp(time=new_times, method='linear')


def ready_to_use_noaa(db: Session, max_hr: int, target_date: str) -> None:
    datasets = []
    global max_predication_hour
    max_predication_hour = max_hr

    gfs_filtered_files = parse_dir_info(target_date,
                                        'gfs',
                                        'gfs.t00z.pgrb2.0p25.f*.grib2_press.csv',
                                        num_index=4,
                                        split_delimiter='.')
    if gfs_filtered_files:
        gfs_result = process_files(gfs_filtered_files, ['pressure', 'water_temp'], target_date)
        gfs_result = gfs_result.drop_vars('date').rename({'pressure': 'pressure', 'water_temp': 'water_temperature'})
        datasets.append(gfs_result)

    hycom_filtered_files = parse_dir_info(target_date,
                                          'hycom',
                                          'hycom_glb_sfc_u_' + target_date[0:8] + '00_t*_current.csv',
                                          num_index=5,
                                          split_delimiter='_')

    if hycom_filtered_files:
        hycom_result = process_files(hycom_filtered_files, ['wather_u', 'wather_v'], target_date)

        # Calculate wind speed and direction
        u, v = hycom_result['wather_u'], hycom_result['wather_v']
        hycom_result['current_speed_real'] = np.sqrt(u ** 2 + v ** 2)
        hycom_result['current_direction_real'] = np.mod(180 + np.arctan2(u, v) * (180 / np.pi), 360)
        hycom_result = hycom_result.drop_vars(['date', 'wather_u', 'wather_v'])
        datasets.append(hycom_result)

    gfswave_filtered_files = parse_dir_info(target_date,
                                            'gfswave',
                                            'gfswave.t00z.global.0p25.f*_wave.csv',
                                            num_index=4,
                                            split_delimiter='.')

    if gfswave_filtered_files:
        gfswave_result = process_files(gfswave_filtered_files, ['dir', 'priod', 'height'], target_date)
        gfswave_result = gfswave_result.drop_vars('date').rename(
            {'dir': 'wave_direction_real', 'priod': 'wave_period', 'height': 'wave_height_real'})
        datasets.append(gfswave_result)

    gfswind_filtered_files = parse_dir_info(target_date,
                                            'gfswave',
                                            'gfswave.t00z.global.0p25.f*_wind.csv',
                                            num_index=4,
                                            split_delimiter='.')
    if gfswind_filtered_files:
        gfswind_result = process_files(gfswind_filtered_files, ['dir', 'speed'], target_date)
        gfswind_result = gfswind_result.drop_vars('date').rename(
            {'dir': 'wind_direction_real', 'speed': 'wind_speed_real'})
        datasets.append(gfswind_result)

    merged_result = xr.merge(datasets)

    # xarray.Dataset을 Pandas DataFrame으로 변환하고 실제 예측 시간(daily) 인서트
    insert_df = merged_result.to_dataframe().reset_index()
    insert_df = insert_df.rename(columns={'time': 'prediction_time'})
    insert_df = insert_df.rename(columns={'lon': 'longitude'})
    insert_df = insert_df.rename(columns={'lat': 'latitude'})

    time_from_target_date = np.datetime64(pd.to_datetime(target_date))

    insert_df['prediction_standard_time'] = time_from_target_date
    insert_df['update_date'] = datetime.now()

    insert_db_from_dataframe(db, insert_df)


def insert_db_from_dataframe(db: Session, df: pd.DataFrame):
    print("bulk_insert started")
    noaa_prediction_hist_service.bulk_upsert_noaa_prediction_hist(db, df)
    print("bulk_insert finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process Noaa forecasts')
    parser.add_argument('-d', '--date', type=str, nargs='+', required=True, help='List of dates in the format YYYYMMDD')
    return parser.parse_args()


def print_land_char():
    shapefile_path = resource_file_path + 'ne_10m_land.shp'
    # Shapefile 읽기
    gdf = gpd.read_file(shapefile_path)
    # 지리 데이터 시각화
    gdf.plot()
    # 이미지 보여주기
    plt.show()


def print_compare_chart(old_filter_df: pd.DataFrame, filter_df: pd.DataFrame, value: str):
    # 시각화 - 원본 데이터와 보간된 데이터 비교
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 원본 데이터 시각화 (좌측)
    scatter = ax[0].scatter(old_filter_df['lon'], old_filter_df['lat'], c=old_filter_df[value], cmap='viridis',
                            edgecolor='k')
    ax[0].set_title('Original Data')
    ax[0].set_xlabel('Longitude')
    ax[0].set_ylabel('Latitude')
    fig.colorbar(scatter, ax=ax[0], label='Value')

    # 2. 보간된 데이터 시각화 (우측)
    contour = ax[1].scatter(filter_df['lon'], filter_df['lat'], c=filter_df[value], cmap='viridis', edgecolor='k')
    ax[1].set_title('Interpolated Values')
    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Latitude')
    fig.colorbar(contour, ax=ax[1], label='Value')

    # 플롯 표시
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     db = next(get_db())
#     ready_to_use_noaa(db, 48, '20240904010000')
