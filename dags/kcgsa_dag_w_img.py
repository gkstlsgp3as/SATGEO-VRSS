import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

host_path = 'http://ship-service.ship.svc.cluster.local:80/'

def w04_api_call(**kwargs):
    conf = kwargs['dag_run'].conf if kwargs['dag_run'] else {}
    print(f"Received conf: {conf}")
    if 'satellite_sar_image_id' not in conf:
        raise KeyError("Missing 'satellite_sar_image_id' key in conf")

    api_path = '/api/risk-mappings/w04'
    api_params = {
        'satellite_sar_image_id': conf['satellite_sar_image_id']
    }
    requests.get(host_path + api_path, params=api_params)


def w05_api_call(**kwargs):
    conf = kwargs['dag_run'].conf if kwargs['dag_run'] else {}
    print(f"Received conf: {conf}")
    if 'satellite_sar_image_id' not in conf:
        raise KeyError("Missing 'satellite_sar_image_id' key in conf")

    api_path = '/api/risk-mappings/w05'
    api_params = {
        'satellite_sar_image_id': conf['satellite_sar_image_id']
    }

    requests.get(host_path + api_path, params=api_params)


with DAG(
        'n_kcgsa_risk-mapping_img_dag',
        default_args={
            'owner': 'airflow',
            'start_date': datetime(2024, 9, 28),
        },
        schedule_interval=None,  # 주기적 실행 없이 API로만 호출
) as dag:

    w04_task = PythonOperator(
        task_id='w04',
        python_callable=w04_api_call,
        provide_context=True
    )

    w05_task = PythonOperator(
        task_id='w05',
        python_callable=w05_api_call,
        provide_context=True
    )

    # api_task
    w04_task
    w05_task
