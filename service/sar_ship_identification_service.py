
from sqlalchemy.orm import Session

from sqlalchemy.dialects.postgresql import insert
from app.models.sar_ship_identification import SarShipIdentification
import pandas as pd

## 데이터 삽입 시 중복 처리
def bulk_upsert_sar_ship_identification(db: Session, bulk_data: pd.DataFrame) -> int:
    stmt = insert(SarShipIdentification).values(bulk_data.to_dict(orient='records'))  
    # pandas.DataFrame을 dict로 변환 후 데이터 삽입 쿼리문 생성 (SQL의 INSERT문)

    # Define ON CONFLICT DO UPDATE
    stmt = stmt.on_conflict_do_update(
        index_elements=['satellite_sar_image_id', 'identification_ship_id'],  # primary key columns
        set_={
            'longitude': stmt.excluded.longitude,
            'latitude': stmt.excluded.latitude,
            'interpolation_cog': stmt.excluded.interpolation_cog,
            'interpolation_sog': stmt.excluded.interpolation_sog,
            'dima_d': stmt.excluded.dima_d,
            'type': stmt.excluded.type,
            'end': stmt.excluded.end,
            'detection_yn': stmt.excluded.detection_yn,
            'detection_longitude': stmt.excluded.detection_longitude,
            'detection_latitude': stmt.excluded.detection_latitude,
        }
    )

    # Execute the statement
    db.execute(stmt)
    db.commit()
