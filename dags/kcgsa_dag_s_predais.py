import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime

host_path = 'http://ship-service.ship.svc.cluster.local:80/'


def s08_api_call(execution_date, **kwargs):
    api_path = '/api/ships/s08'
    api_params = {
        'mmsi': 123456789     # 사용자의 선박 클릭시 전송되는 인자 
        'dates': 2024-12-31 17:29:00
    }

    requests.get(host_path + api_path, params=api_params)


with DAG(
        'n_kcgsa_ship_shipais_dag',
        default_args={
            'owner': 'airflow',
            'start_date': datetime(2024, 12, 29),
        },
        schedule_interval=timedelta(minutes=3)
        #schedule_interval='@hourly',  # 매 시 실행
        catchup=False  # 과거 실행을 건너뛸지 여부 (False로 하면 과거 실행 건너뜀)
) as dag:

    s07_task = PythonOperator(
        task_id='s08',
        python_callable=s08_api_call
    )

    # api_task >> next_task
    s08_task
