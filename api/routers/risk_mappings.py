import datetime
import uuid
from typing import List

from fastapi import APIRouter, Path, Query, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from uvicorn.config import logger

from app.algorithm import w01_collect_ecmwf, w03_ready_to_use_ecmwf, w01_collect_noaa, w03_ready_to_use_noaa, w03_ready_to_use_noaa_cyclone, \
    w08_calculate_navigation_risk, w07_calculate_correction_map
from app.config.db_connection import get_db
from app.models.ecmwf_collect_hist import EcmwfCollectHist
from app.models.noaa_collect_hist import NoaaCollectHist
from app.service import ecmwf_collect_hist_service, noaa_collect_hist_service

router = APIRouter()


@router.get("/w01/ecmwf")
async def collect_ecmwf(background_tasks: BackgroundTasks,
                        dates: List[str] = Query(),
                        task_id: str = Query(...),
                        run_id: str = Query(...),
                        db: Session = Depends(get_db)):
    transaction_id = str(uuid.uuid4()).replace("-", "")
    logger.info(f"====== transaction_id: {transaction_id} ======")
    now = datetime.datetime.now()
    collect_history = EcmwfCollectHist(
        task_id=task_id,
        run_id=run_id,
        transaction_id=transaction_id,
        request_dates=dates,
        status="started",
        created_at=now,
        updated_at=now
    )
    save_result = ecmwf_collect_hist_service.save_ecmwf_collect_history(db, collect_history)

    background_tasks.add_task(w01_collect_ecmwf.lists_by_dates, db, collect_history, dates)
    return save_result


@router.get("/w01/ecmwf/task-status/{transaction_id}")
def get_task_status(transaction_id: str = Path(...),
                    task_id: str = Query(...),
                    run_id: str = Query(...),
                    db: Session = Depends(get_db)):
    return w01_collect_ecmwf.get_task_status(db, task_id, run_id, transaction_id)


@router.get("/w01/noaa")
async def collect_noaa(background_tasks: BackgroundTasks,
                       dates: List[str] = Query(),
                       task_id: str = Query(...),
                       run_id: str = Query(...),
                       db: Session = Depends(get_db)):
    transaction_id = str(uuid.uuid4()).replace("-", "")
    logger.info(f"====== transaction_id: {transaction_id} ======")
    now = datetime.datetime.now()
    collect_history = NoaaCollectHist(
        task_id=task_id,
        run_id=run_id,
        transaction_id=transaction_id,
        request_dates=dates,
        status="started",
        created_at=now,
        updated_at=now
    )
    save_result = noaa_collect_hist_service.save_noaa_collect_history(db, collect_history)

    background_tasks.add_task(w01_collect_noaa.lists_by_dates, db, collect_history, dates)
    return save_result


@router.get("/w01/noaa/task-status/{transaction_id}")
def get_task_status(transaction_id: str = Path(...),
                    task_id: str = Query(...),
                    run_id: str = Query(...),
                    db: Session = Depends(get_db)):
    return w01_collect_noaa.get_task_status(db, task_id, run_id, transaction_id)


@router.delete("/w01/noaa")
async def collect_noaa(date: str = Query()):
    w01_collect_noaa.remove_files(date)


@router.get("/w02")
async def collect_ecmwf_reanalysis(background_tasks: BackgroundTasks,
                       dates: List[str] = Query(),
                       task_id: str = Query(...),
                       run_id: str = Query(...),
                       db: Session = Depends(get_db)):
    transaction_id = str(uuid.uuid4()).replace("-", "")
    logger.info(f"====== transaction_id: {transaction_id} ======")
    now = datetime.datetime.now()
    collect_history = EcmwfReanalysisHist(
        task_id=task_id,
        run_id=run_id,
        transaction_id=transaction_id,
        request_dates=dates,
        status="started",
        created_at=now,
        updated_at=now
    )
    save_result = ecmwf_reanalysis_hist_service.save_ecmwf_reanalysis_history(db, collect_history)

    background_tasks.add_task(w02_collect_ecmwf_reanalysis.lists_by_dates, db, collect_history, dates)
    return save_result


@router.get("/w02/task-status/{transaction_id}")
def get_task_status(transaction_id: str = Path(...),
                    task_id: str = Query(...),
                    run_id: str = Query(...),
                    db: Session = Depends(get_db)):
    return w02_collect_ecmwf_reanalysis.get_task_status(db, task_id, run_id, transaction_id)


@router.get("/w03/ecmwf")
def ready_to_use_ecmwf(max_hr: int = Query(alias="max-hour"),
                       db: Session = Depends(get_db)):
    w03_ready_to_use_ecmwf.ready_to_use_ecmwf(db, max_hr)
    return {"max_hr": max_hr}


@router.get("/w03/noaa")
def ready_to_use_noaa(max_hr: int = Query(alias="max-hour"),
                      date: str = Query(),
                      db: Session = Depends(get_db)):

    logger.info(f"======[noaa] date: {date} ======")
    w03_ready_to_use_noaa.ready_to_use_noaa(db, max_hr, date)
    return {"max_hr": max_hr}


@router.get("/w03/noaa/cyclone")
def ready_to_use_noaa(date_time: str = Query(alias="date-time"),
                      db: Session = Depends(get_db)):

    logger.info(f"======[noaa_cyclone] date_time: {date_time} ======")
    w03_ready_to_use_noaa_cyclone.ready_to_use_noaa_cyclone(db, 48, date_time)
    return {"max_hr": 48}


@router.get("/w04")
async def extract_wind(satellite_sar_image_id: str = Query(alias="sar-img-id"),
                      db: Session = Depends(get_db)):

    logger.info(f"======[noaa_cyclone] satellite_sar_image_id: {satellite_sar_image_id} ======")
    w04_extract_wind.extract_wind(db, satellite_sar_image_id)
    return {"satellite_sar_image_id": satellite_sar_image_id}


@router.get("/w05")
async def extract_wave(satellite_sar_image_id: str = Query(alias="sar-img-id"),
                      db: Session = Depends(get_db)):

    logger.info(f"======[noaa_cyclone] satellite_sar_image_id: {satellite_sar_image_id} ======")
    w04_extract_wave.extract_wave(db, satellite_sar_image_id)
    return {"satellite_sar_image_id": satellite_sar_image_id}


@router.get("/w06")
def update_correction_map(correction_map_id: str = Query(),
                         db: Session = Depends(get_db)):
    w06_update_correction_map.update_correction_map(db, correction_map_id)
    return {"correction_map_id": correction_map_id}


@router.get("/w07")
def calculate_correction(correction_map_id: str = Query(),
                         db: Session = Depends(get_db)):
    w07_calculate_correction_map.calculate_correction(db, correction_map_id)
    return {"correction_map_id": correction_map_id}


@router.get("/w08")
def calculate_navigation_risk(voyage_risk_map_id: str = Query(),
                              db: Session = Depends(get_db)):
    w08_calculate_navigation_risk.calculate_navigation_risk(db, voyage_risk_map_id)
    return {"voyage_risk_map_id": voyage_risk_map_id}
