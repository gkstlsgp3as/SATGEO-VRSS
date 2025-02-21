import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.noaa_prediction_hist import NoaaPredictionHist


def bulk_upsert_noaa_prediction_hist(db: Session, data: pd.DataFrame) -> int:
    stmt = insert(NoaaPredictionHist).values(data.to_dict(orient='records'))

    # ON CONFLICT 구문을 사용하여 중복되는 경우 업데이트 처리
    stmt = stmt.on_conflict_do_update(
        index_elements=['prediction_standard_time', 'prediction_time', 'latitude', 'longitude'],  # 중복 여부를 판단할 컬럼 (예: id)
        set_={
            'pressure': stmt.excluded.pressure,  # 업데이트할 컬럼들
            'water_temperature': stmt.excluded.water_temperature,
            'current_speed_real': stmt.excluded.current_speed_real,
            'current_direction_real': stmt.excluded.current_direction_real,
            'wave_direction_real': stmt.excluded.wave_direction_real,
            'wave_period': stmt.excluded.wave_period,
            'wave_height_real': stmt.excluded.wave_height_real,
            'wind_direction_real': stmt.excluded.wind_direction_real,
            'wind_speed_real': stmt.excluded.wind_speed_real,
            'update_date': stmt.excluded.update_date,
            # 필요에 따라 다른 컬럼도 추가
        }
    )

    # DB 세션에서 실행
    db.execute(stmt)
    db.commit()
