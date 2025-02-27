# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:28:44 2025

@author: user

# home 디렉토리에 .cdsapirc 파일 추가 후 url과 key를 붙여넣기
# url: https://cds.climate.copernicus.eu/api
# key: <PERSONAL-ACCESS-TOKEN>  
# --> key는 개인 계정페이지에서 복사

# 참고자료: https://cds.climate.copernicus.eu/how-to-api

# relative_humidity는 재분석 자료에서 못찾음
# 기타 변수는 아래 들어간 이름 그대로 존재 함. 

"""

import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_wave_direction",
        "significant_height_of_combined_wind_waves_and_swell",
        "total_precipitation",
        "skin_temperature",
        "total_column_water_vapour"
    ],
    "year": ["2024", "2025"],
    "month": ["01", "02", "12"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()






















