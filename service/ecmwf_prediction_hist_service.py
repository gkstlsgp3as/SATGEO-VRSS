import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from app.models.ecmwf_prediction_hist import EcmwfPredictionHist


def bulk_insert_ecmwf_prediction_history(db: Session, bulk_data: pd.DataFrame):
    try:
        db.bulk_insert_mappings(EcmwfPredictionHist, bulk_data.to_dict(orient='records'))
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()


def bulk_upsert_ecmwf_prediction_hist(db: Session, data: pd.DataFrame) -> int:
    stmt = insert(EcmwfPredictionHist).values(data.to_dict(orient='records'))

    # ON CONFLICT 구문을 사용하여 중복되는 경우 업데이트 처리
    stmt = stmt.on_conflict_do_update(
        index_elements=['prediction_standard_time', 'prediction_time', 'latitude', 'longitude'],  # 중복 여부를 판단할 컬럼 (예: id)
        set_={
            'wind_speed_real': stmt.excluded.wind_speed_real,  # 업데이트할 컬럼들
            'wind_direction_real': stmt.excluded.wind_direction_real,
            'wave_height_real': stmt.excluded.wave_height_real,
            'wave_direction_real': stmt.excluded.wave_direction_real,
            'relative_humidity': stmt.excluded.relative_humidity,
            'total_column_water_vapor': stmt.excluded.total_column_water_vapor,
            'skin_temperature': stmt.excluded.skin_temperature,
            # 필요에 따라 다른 컬럼도 추가
        }
    )

    # DB 세션에서 실행
    db.execute(stmt)
    db.commit()
