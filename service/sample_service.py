
from sqlalchemy.orm import Session

from app.models.모델_파일 import 모델_클래스

# --------------------------------------------------------------------------------------------
## 데이터 조회
def get_sample(db: Session, id: str) -> List[모델_클래스]:
    return db.query(모델_클래스).filter(모델_클래스.id == id).all()

## 데이터 조회 - 여러 필터
def get_sample(db: Session,  id1: str, id2: str, id3: str) -> List[모델_클래스]:
    return db.query(모델_클래스).filter(
                                  모델_클래스.id1 == id1,
                                  모델_클래스.id2 == id2,
                                  모델_클래스.id3 == id3,
                                  ).first()

# --------------------------------------------------------------------------------------------
## 단일 데이터 삽입
def save_sample(db: Session, sample_model_class: SampleModelClass) -> SampleModelClass:
    db.add(sample_model_class)
    db.commit()
    db.refresh(sample_model_class)
    logger.info("====== saved sample_model_class ======")
    return sample_model_class

## 여러 데이터 삽입; 딕셔너리 형태의 데이터, add보다 빠름
def bulk_insert_sample(db: Session, bulk_data: pd.DataFrame):
    try:
        db.bulk_insert_mappings(모델_클래스, bulk_data.to_dict(orient='records'))
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

# --------------------------------------------------------------------------------------------
## 데이터 삽입 시 중복 처리
from sqlalchemy.dialects.postgresql import insert
from app.models.sample_model_class import SampleModelClass

def bulk_upsert_sample(db: Session, bulk_data: pd.DataFrame) -> int:
    stmt = insert(SampleModelClass).values(bulk_data.to_dict(orient='records'))  
    # pandas.DataFrame을 dict로 변환 후 데이터 삽입 쿼리문 생성 (SQL의 INSERT문)

    # ON CONFLICT 구문을 사용하여 중복되는 경우 업데이트 처리
    stmt = stmt.on_conflict_do_update(
        index_elements=['field1', 'field2'],  # 중복 여부를 판단할 컬럼 (예: id)
        set_={
            'field3': stmt.excluded.field3,  # 중복 시 업데이트할 컬럼들; excluded = 새로 삽입한(충돌한) 데이터
            'field4': stmt.excluded.field4,  # 업데이트 하지 않을 컬럼은 포함x 
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

# --------------------------------------------------------------------------------------------
## 조회 후 데이터 일부 변경 
def update_sample(db: Session, sample_model_class: SampleModelClass) -> SampleModelClass:
    sample_model_class = db.query(SampleModelClass).filter(SampleModelClass.id == sample_model_class.id).first()

    sample_model_class.status = noaa_collect_history.status
    sample_model_class.updated_at = noaa_collect_history.updated_at
    db.commit()
    db.refresh(sample_model_class)
    logger.info("====== updated sample_model_class ======")
    return sample_model_class
