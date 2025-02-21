from fastapi import APIRouter
from app.config.settings import settings  # 필요한 파라미터 정의 
from app.config.db_connection import get_db

# 해당하는 알고리즘, DB모델, DB서비스 import 
from app.algorithm import s01_detect_ships, s02_estimate_ship_velocity, s03_correct_ais_time, s04_find_unidentified_ships, \
                          s05_classify_unidentified_ships, s06_predict_routes_ship, s07_predict_routes_ais, \
                          s08_generate_ais_distribution, s09_find_l1a, s10_predict_routes_mtn, \
                          s11_calculate_ship_navigation_risk, s12_generate_ship_reports, s13_generate_area_risk_reports
from app.models.sar_ship_identification import SarShipIdentification
from app.models.sar_ship_unidentification import SarShipUnidentification
from app.models.ship_prediction_route import ShipPredictionRoute
from app.models.ship_voyage_risk_map import ShipVoyageRiskMap

from app.service import sar_ship_identification_service, sar_ship_unidentification_service, \
                        ship_prediction_route_service, ship_voyage_risk_map_service, \
                        ship_route_safety_report_service, area_route_safety_report_service

## TODO
# 1. 인자 처리 #1: 디렉토리 관련한 인자는 모두 config/settings에 정의하여 algorithm/ 하위 파일에 할당
# 예) input_dir = settings.W01_OUTPUT_PATH 예2) geojson = settings.W03_RESOURCE_PATH + "/map.geojson"
# 2. 인자 처리 #2: 그 외 인자만 남겨두고 Query로 받도록 수정 -> 더 자세한 사항은 README.md 파일 참조 
# 예) max_hr: int = Query(alias="max-hour")
# 3. 인자 처리 #3: db: Session = Depends(get_db) 추가 -> 테이블 접근할 수 있는 DB 연결 
# 4. 함수 호출: algorithm/ 하위의 파일 호출 
# 예) w03_ready_to_use_ecmwf.ready_to_use_ecmwf(db, max_hr)
# 5. 리턴값 지정: 로그에서 확인하고 싶은 값 리턴 혹은 wait를 확인하기 위해 필요한 값 리턴
# 예) return {"max_hr": max_hr}

router = APIRouter()

@router.get("/s01")
def detect_ships(satellite_sar_image_id: str = Query(...),
                      db: Session = Depends(get_db)):

    logger.info(f"======[detect_ships] detect_ships ~~ ======")
    s01_detect_ships.detect_ships(db, satellite_sar_image_id)
    #if status == 'failed':
    #    return {"status": status}
    logger.info(f"======[detect_ships] correct_ais_time ~~ ======")
    s03_correct_ais_time.correct_ais_time(db, satellite_sar_image_id)
    #if status == 'failed':
    #    return {"status": status}
    logger.info(f"======[detect_ships] find_unidentified_ships ~~ ======")
    status = s04_find_unidentified_ships.find_unidentified_ships(db, satellite_sar_image_id)
    return {"status": status} # 미식별 선박이 없는 경우, 'no_unidentified' return해서 S02 실행하지 않도록; 미식별선박 존재하는 경우, 'completed' 


@router.get("/s02")
def estimate_ship_velocity(satellite_sar_image_id: str = Query(),
                         db: Session = Depends(get_db)):
    logger.info(f"======[estimate_ship_velocity] find_l1a ~~ ======")
    s09_find_l1a.find_l1a(db, satellite_sar_image_id)
                           
    logger.info(f"======[estimate_ship_velocity] estimate_ship_velocity ~~ ======")
    s02_estimate_ship_velocity.estimate_ship_velocity(db, satellite_sar_image_id)

    logger.info(f"======[estimate_ship_velocity] classify_unidentified_ships ~~ ======")
    s05_classify_unidentified_ships.classify_unidentified_ships(db, satellite_sar_image_id)

    return {"satellite_sar_image_id": satellite_sar_image_id}


@router.get("/s06")
def s06_predict_routes_ship(ship_id: str = Query(),
                              db: Session = Depends(get_db)):
    s06_predict_routes_ship.predict_routes_ship(db, interest_ship_id)
    return {"ship_id": ship_id}

@router.get("/s07")
def s07_predict_routes_ais(ship_id: str = Query(),
                           request_time = Query(), 
                           standard_prediction_time = Query(),
                              db: Session = Depends(get_db)):
    s07_predict_routes_ais.predict_routes_ais(db, request_time, standard_prediction_time, ship_id)
    return {"ship_id": ship_id}

@router.get("/s08")
def generate_ais_distribution(date: str = Query(),
                              db: Session = Depends(get_db)):
    s08_generate_ais_distribution.generate_ais_distribution(db, date)
    return {"date": date}


@router.get("/s10")
def predict_routes_mtn(interest_ship_id: str = Query(),
                              db: Session = Depends(get_db)):
    s10_predict_routes_mtn.predict_routes_mtn(db, interest_ship_id)
    return {"interest_ship_id": interest_ship_id}


@router.get("/s11")
def calculate_ship_navigation_risk(voyage_risk_map_id: str = Query(),
                                        mmsi_process: str = Query(),
                                       db: Session = Depends(get_db)):
    s11_calculate_ship_navigation_risk.calculate_ship_navigation_risk(db, voyage_risk_map_id, mmsi_process)
    return {"voyage_risk_map_id": voyage_risk_map_id, "mmsi_process": mmsi_process}


@router.get("/s12")
def generate_ship_risk_reports(interest_ship_id: str = Query(),
                              db: Session = Depends(get_db)):
    s12_generate_ship_risk_reports.generate_ship_risk_reports(db, interest_ship_id)
    return {"interest_ship_id": interest_ship_id}

@router.get("/s13")
def generate_area_risk_reports(interest_area_id: str = Query(),
                              db: Session = Depends(get_db)):
    s13_generate_area_risk_reports.generate_area_risk_reports(db, interest_area_id)
    return {"interest_area_id": interest_area_id}
