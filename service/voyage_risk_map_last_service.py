import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.voyage_risk_map_last import VoyageRiskMapLast

def update_voyage_risk_map_last(db: Session, bulk_data: pd.DataFrame) -> int:
    stmt = insert(VoyageRiskMapLast).values(bulk_data.to_dict(orient='records'))

    # ON CONFLICT 구문을 사용하여 중복되는 경우 업데이트 처리
    stmt = stmt.on_conflict_do_update(
        index_elements=['longitude', 'latitude'],  # 중복 여부를 판단할 컬럼 (예: id)
        set_={
            'voyage_risk_map_id': stmt.excluded.voyage_risk_map_id,  # 업데이트할 컬럼들
            'longitude_length': stmt.excluded.longitude_length,
            'latitude_length': stmt.excluded.latitude_length,
            'wind_speed': stmt.excluded.wind_speed,
            'wind_direction': stmt.excluded.wind_direction,
            'wave_height': stmt.excluded.wave_height,
            'wave_direction': stmt.excluded.wave_direction,
            'relative_humidity': stmt.excluded.relative_humidity,
            'total_column_water_vapor': stmt.excluded.total_column_water_vapor,
            'skin_temperature': stmt.excluded.skin_temperature,
            'tidal_current_speed': stmt.excluded.tidal_current_speed,
            'tidal_current_direction': stmt.excluded.tidal_current_direction,
            'typhoon_longitude': stmt.excluded.typhoon_longitude,
            'typhoon_latitude': stmt.excluded.typhoon_latitude,
            'typhoon_grade': stmt.excluded.typhoon_grade,
        }
    )

    # DB 세션에서 실행
    db.execute(stmt)
    db.commit()
