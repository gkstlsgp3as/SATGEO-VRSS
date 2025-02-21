import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime

host_path = 'http://ship-service.ship.svc.cluster.local:80/'
# host_path = 'http://127.0.0.1:8000'
# host_path = 'http://host.docker.internal:8000'

def s01_api_call(execution_date, **kwargs):
    # 트리거시킨 DAG run에서 전달된 conf 파라미터 접근
    # ({"satellite_sar_image_id": "123456789"} 로 전달한다고 가정)
    conf = kwargs['dag_run'].conf if kwargs['dag_run'] else {}
    print(f"Received conf: {conf}")
    if 'satellite_sar_image_id' not in conf:
        raise KeyError("Missing 'satellite_sar_image_id' key in conf")

    api_path = '/api/risk-mappings/s01'
    api_params = {
        'satellite_sar_image_id': conf['satellite_sar_image_id'],
    }
    response = requests.get(host_path + api_path, params=api_params)

    if response.status == 'completed':    # 미식별 선박이 존재하면
        kwargs['ti'].xcom_push(key='s01_img_id', value=api_params)
        return 's02_task'
    else:    # status == 'no_unidentification'
        print(f"Note: No Unidentifcation Ships")


def s02_api_call(execution_date, **kwargs):
    api_path = '/api/risk-mappings/s02'
    api_params = {
        'satellite_sar_image_id': kwargs['ti'].xcom_pull(key='s01_img_id')
    }
    requests.get(host_path + api_path, params=api_params)


with DAG(
        'api_trigger',
        default_args={
            'owner': 'airflow',
            'start_date': datetime(2025, 2, 1)
        },
        schedule_interval=None,  # 주기적 실행 없이 API로만 호출
) as dag:

    s01_task = BranchPythonOperator(
        task_id='s01',
        python_callable=s01_api_call,
        provide_context=True
    )

    s02_task = PythonOperator(
        task_id='s02',
        python_callable=s02_api_call
    )

    s01_task >> s02_task
