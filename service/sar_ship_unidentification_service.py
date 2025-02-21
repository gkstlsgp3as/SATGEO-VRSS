
from sqlalchemy.orm import Session

from sqlalchemy.dialects.postgresql import insert
from app.models.sar_ship_unidentification import SarShipUnidentification

# --------------------------------------------------------------------------------------------
## 데이터 조회
def get_sar_ship_unidentification(db: Session, satellite_sar_image_id: str) -> List[SarShipUnidentification]:
    return db.query(SarShipUnidentification).filter(SarShipUnidentification.satellite_sar_image_id == satellite_sar_image_id).all()

# --------------------------------------------------------------------------------------------
## 데이터 삽입 시 중복 처리
def bulk_upsert_sar_ship_unidentification(db: Session, bulk_data: pd.DataFrame) -> int:
    stmt = insert(SarShipUnidentification).values(bulk_data.to_dict(orient='records'))  
    # pandas.DataFrame을 dict로 변환 후 데이터 삽입 쿼리문 생성 (SQL의 INSERT문)

    # ON CONFLICT 구문을 사용하여 중복되는 경우 업데이트 처리
    stmt = stmt.on_conflict_do_update(
        index_elements=['satellite_sar_image_id', 'unidentification_ship_id'],  # 중복 여부를 판단할 컬럼 (예: id)
        set_={
            'prediction_ship_type': stmt.excluded.prediction_ship_type,  # 중복 시 업데이트할 컬럼들; excluded = 새로 삽입한(충돌한) 데이터
        }
    )

    try:
        db.execute(stmt)  # SQL 쿼리문 실행
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()
