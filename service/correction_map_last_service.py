from typing import List
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.correction_map_last import CorrectionMapLast


def update_correction_map_last(db: Session, bulk_data: pd.DataFrame) -> int:
    stmt = insert(CorrectionMapLast).values(bulk_data.to_dict(orient='records'))

    # ON CONFLICT 구문을 사용하여 중복되는 경우 업데이트 처리
    stmt = stmt.on_conflict_do_update(
        index_elements=['longitude', 'latitude'],  # 중복 여부를 판단할 컬럼 (예: id)
        set_={
            'correction_map_id': stmt.excluded.correction_map_id,  # 업데이트할 컬럼들
            'longitude_length': stmt.excluded.longitude_length,
            'latitude_length': stmt.excluded.latitude_length,
            'wind_speed': stmt.excluded.wind_speed,
            'wind_direction': stmt.excluded.wind_direction,
            'wave_height': stmt.excluded.wave_height,
            'wave_direction': stmt.excluded.wave_direction,
            'relative_humidity': stmt.excluded.relative_humidity,
            'total_column_water_vapor': stmt.excluded.total_column_water_vapor,
            'skin_temperature': stmt.excluded.skin_temperature,
        }
    )

    # DB 세션에서 실행
    # db.execute(stmt)
    # db.commit()

    try:
        db.execute(stmt)
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()
