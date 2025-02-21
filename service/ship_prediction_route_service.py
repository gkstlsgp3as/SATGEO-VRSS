from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List
from models import AisShipDataHist  # 테이블 클래스 임포트

from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from models import ShipPredictionRoute  # 테이블 클래스 임포트
from utils.cfg import Cfg
from datetime import datetime

def get_ship_data(db: Session, datetime: datetime) -> List[AisShipDataHist]:
    """
    AIS_SHIP_DATA_HIST 테이블에서 특정 datetime으로부터 previous_time 이전 데이터를 조회합니다.

    Args:
        db (Session): 데이터베이스 세션
        datetime (datetime): 기준 시간
        previous_time (int): 기준 시간으로부터 이전 시간 (시간 단위)

    Returns:
        List[AisShipDataHist]: 조회된 데이터 리스트
    """
    start_time = datetime - timedelta(hours=Cfg.previous_time)
    return db.query(AisShipDataHist)\
        .filter(AisShipDataHist.timestamp >= start_time, AisShipDataHist.timestamp <= datetime)\
        .all()


def save_ship_prediction_route(db: Session, save_data: dict):
    """
    예측 데이터를 SHIP_PREDICTION_ROUTE 테이블에 추가합니다.
    중복 데이터는 업데이트 처리합니다.

    Args:
        db (Session): 데이터베이스 세션
        save_data (dict): 저장할 데이터 딕셔너리
        'mmsi': mmsi,
        'request_time': current_time,
        'target_time': target_time,
        'target_lon': source_data['lon'][-1], # 추후에 시간 맞춰 수정
        'target_lat': source_data['lat'][-1], # 추후에 시간 맞춰 수정
        'arrive_lon': predict_data['lon'][-1], 
        'arrive_lat': predict_data['lat'][-1],
        'require_time': start_time - finish_time, # 추후에 시간 맞춰 수정
        'distance': 0.0, # 추후에 산정하여 추가
        'predict_route': predict_route,
        'predict_circle': 3.14  # 예시 값, 확률반경 알고리즘 추후 추가

    """
    try:
        # Data 준비: pandas DataFrame으로 변환
        bulk_data = pd.DataFrame([{
            'ship_id': save_data['mmsi'],  # 선박 ID
            'request_time': save_data['request_time'],  # 예측 시간
            'prediction_route_type': 'example_type',  # 예시 값, 필요에 따라 수정
            'standard_prediction_time': save_data['target_time'],  # 예측 기준 시간
            'start_longitude': save_data['target_lon'],  # 선택한 선박 경도
            'start_latitude': save_data['target_lat'],  # 선택한 선박 위도
            'arrival_longitude': save_data['arrive_lon'],  # 마지막 시간 경로 (필요하면 계산)
            'arrival_latitude': save_data['arrive_lat'],  # 도착 위도 (필요하면 계산)
            'rp_type': 1,  # 예시 값, 필요에 따라 수정
            'rp_requirement_second': save_data['require_time'],  # 함수 실행 시간
            'route_distance': save_data['distance'],  # 예측 경로 거리
            'route_requirement_second': 0.0,  # MTN에서 사용. 
            'route': save_data['predict_route'],  # 예측 경로
            'route_geom': None,  # 필요하면 Geometry 데이터 삽입
            'predict_circle': 3.14  # 예시 값, 확률반경 알고리즘 추후 추가
        }])

        # UPSERT 쿼리 작성
        stmt = insert(ShipPredictionRoute).values(bulk_data.to_dict(orient='records'))
        stmt = stmt.on_conflict_do_update(
            index_elements=['ship_id', 'request_time', 'prediction_route_type'],
            set_={
                'start_longitude': stmt.excluded.start_longitude,
                'start_latitude': stmt.excluded.start_latitude,
                'standard_prediction_time': stmt.excluded.standard_prediction_time
            }
        )
        # 데이터가 이미 존재하면, 해당 데이터 갱신, 없으면 insert

        # 실행
        db.execute(stmt)
        db.commit()

    except Exception as e:
        print(f"An error occurred during upsert: {e}")
        db.rollback()
    finally:
        db.close()
