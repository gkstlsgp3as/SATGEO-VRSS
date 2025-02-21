from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from app.models.correction_map_hist import CorrectionMapHist


def get_correction_map_history(db: Session, correction_map_id: str) -> List[CorrectionMapHist]:
    return db.query(CorrectionMapHist).filter(CorrectionMapHist.correction_map_id == correction_map_id).all()


def bulk_insert_correction_map_history(db: Session, bulk_data: pd.DataFrame):
    try:
        db.bulk_insert_mappings(CorrectionMapHist, bulk_data.to_dict(orient='records'))
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()
