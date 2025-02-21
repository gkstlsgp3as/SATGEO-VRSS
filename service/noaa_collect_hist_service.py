import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session
from uvicorn.config import logger

from app.models.noaa_collect_hist import NoaaCollectHist


def get_noaa_collect_history(db: Session, task_id: str, run_id: str, transaction_id: str) -> Optional[NoaaCollectHist]:
    collect_history = db.query(NoaaCollectHist)\
        .filter(
            NoaaCollectHist.task_id == task_id,
            NoaaCollectHist.run_id == run_id,
            NoaaCollectHist.transaction_id == transaction_id
        )\
        .first()
    if collect_history is None:
        return None
    collect_history.try_number += 1
    collect_history.updated_at = datetime.datetime.now()
    db.commit()
    db.refresh(collect_history)
    logger.info("====== updated noaa_collect_history ======")
    return collect_history


def save_noaa_collect_history(db: Session, noaa_collect_history: NoaaCollectHist) -> NoaaCollectHist:
    db.add(noaa_collect_history)
    db.commit()
    db.refresh(noaa_collect_history)
    logger.info("====== saved noaa_collect_history ======")
    return noaa_collect_history


def update_noaa_collect_history(db: Session, noaa_collect_history: NoaaCollectHist) -> NoaaCollectHist:
    collect_history = db.query(NoaaCollectHist).filter(NoaaCollectHist.transaction_id == noaa_collect_history.transaction_id).first()
    if collect_history is None:
        raise HTTPException(status_code=404, detail="not found")
    collect_history.status = noaa_collect_history.status
    collect_history.updated_at = noaa_collect_history.updated_at
    db.commit()
    db.refresh(collect_history)
    logger.info("====== updated noaa_collect_history ======")
    return noaa_collect_history
