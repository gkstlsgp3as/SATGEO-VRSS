import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.service import noaa_cyclone_prediction_hist_service

from app.config.db_connection import get_db

local_file_path = settings.W01_NOAA_OUTPUT_PATH + "/"
resource_file_path = settings.W03_NOAA_RESOURCE_PATH + "/"
batch_size = 1000
max_predication_hour = 48  # default


def parse_cyclone_dir_info(target_date: str, sub_path: str, filename_pattern: str, num_index: int, split_delimiter: str) -> str:
    full_path = os.path.join(local_file_path, target_date[0:8], sub_path)
    all_files = glob.glob(os.path.join(full_path, filename_pattern))

    filtered_file = ''
    maximum = 0
    for file in all_files:
        base_name = os.path.basename(file)
        number_str = base_name.split(split_delimiter)[num_index][8:10]  # 파일명에서 특정 인덱스 추출

        if maximum <= int(number_str):
            filtered_file = file

    print(filtered_file)
    return filtered_file


def process_cyclone_files(file_path: str, target_date: str) -> List:
    # XML 파싱
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_data = file.read()

    root = ET.fromstring(xml_data)

    # 모든 disturbance 요소를 추출
    disturbances = []
    for disturbance in root.findall('.//disturbance'):
        disturbance_data = parse_disturbance(disturbance)
        disturbances.append(disturbance_data)

    # # 파싱한 데이터 출력
    # for disturbance in disturbances:
    #     print(f"Cyclone Name: {disturbance['cyclone_name']}")
    #     print(f"Cyclone Number: {disturbance['cyclone_number']}")
    #     print(f"Basin: {disturbance['basin']}")
    #     for fix in disturbance['fixes']:
    #         print(f"  - Time: {fix['valid_time']}, Lat: {fix['latitude']}, Lon: {fix['longitude']}, Pressure: {fix['pressure']} hPa, Wind Speed: {fix['wind_speed']} kt")

    df_disturbance_interpolated = []
    for disturbance in disturbances:
        result_df = pd.DataFrame()
        for fix in disturbance['fixes']:
            if fix['fix_hour'] <= 48:
                fix['valid_time'] = pd.to_datetime(fix['valid_time'])
                filtered_df = pd.DataFrame([fix])
                result_df = pd.concat([result_df, filtered_df], ignore_index=True)

        # interpolation (temporal)
        result_df.set_index('valid_time', inplace=True)
        df_hourly = result_df.resample('h').asfreq()  # 매 시간 간격으로 리샘플링
        df_hourly_interpolated = df_hourly.interpolate(method='linear')

        df_disturbance_interpolated.append(df_hourly_interpolated)
        # print(df_disturbance_interpolated)

    return df_disturbance_interpolated

# Disturbance 정보를 파싱하는 함수
def parse_disturbance(disturbance_element):
    cyclone_name = disturbance_element.find('cycloneName').text
    cyclone_number = disturbance_element.find('cycloneNumber').text
    basin = disturbance_element.find('basin').text

    # 각 fix 태그들을 순회하며 정보 추출
    fixes = []
    for fix in disturbance_element.findall('fix'):
        fix_data = parse_cyclone(fix)
        fixes.append(fix_data)

    return {
        'cyclone_name': cyclone_name,
        'cyclone_number': cyclone_number,
        'basin': basin,
        'fixes': fixes
    }


# Cyclone 정보를 파싱하는 함수
def parse_cyclone(fix_element):
    fix_hour = fix_element.attrib['hour']
    valid_time = fix_element.find('validTime').text
    latitude = fix_element.find('latitude').text
    longitude = fix_element.find('longitude').text
    pressure = fix_element.find('.//minimumPressure/pressure').text
    wind_speed = fix_element.find('.//maximumWind/speed').text

    return {
        'fix_hour': int(fix_hour),
        'valid_time': valid_time,
        'latitude': float(latitude),
        'longitude': float(longitude),
        'pressure': float(pressure),
        'wind_speed': float(wind_speed)
    }


def ready_to_use_noaa_cyclone(db: Session, max_hr: int, target_date: str) -> None:
    noaa_cyclone_filtered_file = parse_cyclone_dir_info(target_date,
                                                        'noaa_cyclone',
                                                        'kwbc_*_GFS_glob_prod_sttr_glo.xml',
                                                        num_index=1,
                                                        split_delimiter='_')

    noaa_cyclone_result = process_cyclone_files(noaa_cyclone_filtered_file, target_date)
    for insert_df in noaa_cyclone_result:
        insert_df = insert_df.reset_index()
        insert_df['prediction_standard_time'] = pd.to_datetime(target_date)
        insert_df = insert_df.rename(columns={'valid_time': 'prediction_time'})
        insert_df = insert_df.rename(columns={'wind_speed': 'wind_speed_real'})
        insert_df['update_date'] = datetime.now()
        insert_df.drop('fix_hour', axis=1, inplace=True)

        insert_db_from_dataframe(db, insert_df)


def insert_db_from_dataframe(db: Session, df: pd.DataFrame):
    print("bulk_insert started")
    noaa_cyclone_prediction_hist_service.bulk_upsert_noaa_cyclone_prediction_hist(db, df)
    print("bulk_insert finished")


# if __name__ == "__main__":
#     db = next(get_db())
#     ready_to_use_noaa_cyclone(db, 48, '20240904060000')
