
from sqlalchemy.orm import Session

from sqlalchemy.dialects.postgresql import insert
from app.models.sar_ship_unidentification import SarShipUnidentification

import pandas as pd

## 데이터 삽입 시 중복 처리
def bulk_upsert_sar_ship_unidentification(db: Session, bulk_data: pd.DataFrame) -> int:
    stmt = insert(SarShipUnidentification).values(bulk_data.to_dict(orient='records'))  
    # pandas.DataFrame을 dict로 변환 후 데이터 삽입 쿼리문 생성 (SQL의 INSERT문)

    # Define ON CONFLICT DO UPDATE
    stmt = stmt.on_conflict_do_update(
        index_elements=['satellite_sar_image_id', 'unidentification_ship_id'],  # primary key columns
        set_={
            'longitude': stmt.excluded.longitude,
            'latitude': stmt.excluded.latitude,
            'prediction_length': stmt.excluded.prediction_length,
            'prediction_width': stmt.excluded.prediction_width
        }
    )

    # Execute the statement
    db.execute(stmt)
    db.commit()
    
    
def bulk_upsert_sar_ship_unidentification_velocity(db: Session, data: pd.DataFrame) -> int:
    # Prepare insert statement
    stmt = insert(SarShipUnidentification).values(data.to_dict(orient='records'))

    # Define ON CONFLICT DO UPDATE for specific fields
    stmt = stmt.on_conflict_do_update(
        index_elements=['satellite_sar_image_id', 'identification_ship_id'],  # primary key columns
        set_={
            'prediction_cog': stmt.excluded.prediction_cog,
            'prediction_sog': stmt.excluded.prediction_sog
            # Add other fields if conditions apply, or you want more control over what gets updated.
        }
    )

    # Execute the statement
    db.execute(stmt)
    db.commit()
    
def bulk_upsert_sar_ship_unidentification_type(db: Session, data: pd.DataFrame) -> int:
    # Prepare insert statement
    stmt = insert(SarShipUnidentification).values(data.to_dict(orient='records'))

    # Define ON CONFLICT DO UPDATE for specific fields
    stmt = stmt.on_conflict_do_update(
        index_elements=['satellite_sar_image_id', 'identification_ship_id'],  # primary key columns
        set_={
            'prediction_ship_type': stmt.excluded.prediction_ship_type,
            # Add other fields if conditions apply, or you want more control over what gets updated.
        }
    )

    # Execute the statement
    db.execute(stmt)
    db.commit()