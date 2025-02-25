from sqlalchemy.orm import Session
import numpy as np
import pandas as pd

from app.service import correction_map_hist_service, correction_map_last_service


def update_correction(db: Session, correction_map_id: str):
    # 범위 설정
    longitude_range = np.arange(120, 140.25, 0.25)  # 120부터 140까지 0.25 간격
    latitude_range = np.arange(25, 45.25, 0.25)  # 25부터 45까지 0.25 간격

    # 가능한 모든 (longitude, latitude) 조합 생성
    longitude_latitude_grid = [(lon, lat) for lon in longitude_range for lat in latitude_range]

    # 데이터프레임 생성
    data = {
        'correction_map_id': correction_map_id,
        'longitude': [lon for lon, lat in longitude_latitude_grid],
        'latitude': [lat for lon, lat in longitude_latitude_grid],
        'longitude_length': np.random.uniform(0.1, 1.0, len(longitude_latitude_grid)),  # 랜덤 값 생성
        'latitude_length': np.random.uniform(0.1, 1.0, len(longitude_latitude_grid)),
        'wind_speed': np.random.uniform(0, 30, len(longitude_latitude_grid)),
        'wind_direction': np.random.uniform(0, 360, len(longitude_latitude_grid)),
        'wave_height': np.random.uniform(0, 10, len(longitude_latitude_grid)),
        'wave_direction': np.random.uniform(0, 360, len(longitude_latitude_grid)),
        'relative_humidity': np.random.uniform(50, 100, len(longitude_latitude_grid)),
        'total_column_water_vapor': np.random.uniform(10, 50, len(longitude_latitude_grid)),
        'skin_temperature': np.random.uniform(15, 35, len(longitude_latitude_grid))
    }

    # DataFrame 생성
    df = pd.DataFrame(data)

    # bulk insert
    correction_map_hist_service.bulk_insert_correction_map_history(db, df)

    # bulk upsert
    correction_map_last_service.update_correction_map_last(db, df)
