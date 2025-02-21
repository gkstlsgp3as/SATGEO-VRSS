from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from app.models.voyage_risk_map_hist import VoyageRiskMapHist


def get_voyage_risk_map_history(db: Session, voyage_risk_map_id: str) -> List[VoyageRiskMapHist]:
    return db.query(VoyageRiskMapHist).filter(VoyageRiskMapHist.voyage_risk_map_id == voyage_risk_map_id).all()


def bulk_insert_voyage_risk_map_history(db: Session, bulk_data: pd.DataFrame) -> int:
    try:
        db.bulk_insert_mappings(VoyageRiskMapHist, bulk_data.to_dict(orient='records'))
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()
