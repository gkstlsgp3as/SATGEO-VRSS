import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session
from uvicorn.config import logger

from app.models.ecmwf_collect_hist import EcmwfCollectHist


def get_ecmwf_collect_history(db: Session, transaction_id: str) -> Optional[EcmwfCollectHist]:
    collect_history = db.query(EcmwfCollectHist).filter(EcmwfCollectHist.transaction_id == transaction_id).first()
    if collect_history is None:
        return None
    collect_history.try_number += 1
    collect_history.updated_at = datetime.datetime.now()
    db.commit()
    db.refresh(collect_history)
    logger.info("====== updated ecmwf_collect_history ======")
    return collect_history


def save_ecmwf_collect_history(db: Session, ecmwf_collect_history: EcmwfCollectHist) -> EcmwfCollectHist:
    db.add(ecmwf_collect_history)
    db.commit()
    db.refresh(ecmwf_collect_history)
    logger.info("====== saved ecmwf_collect_history ======")
    return ecmwf_collect_history


def update_ecmwf_collect_history(db: Session, ecmwf_collect_history: EcmwfCollectHist) -> EcmwfCollectHist:
    collect_history = db.query(EcmwfCollectHist).filter(EcmwfCollectHist.transaction_id == ecmwf_collect_history.transaction_id).first()
    if collect_history is None:
        raise HTTPException(status_code=404, detail="not found")
    collect_history.status = ecmwf_collect_history.status
    collect_history.updated_at = ecmwf_collect_history.updated_at
    db.commit()
    db.refresh(collect_history)
    logger.info("====== updated ecmwf_collect_history ======")
    return ecmwf_collect_history
