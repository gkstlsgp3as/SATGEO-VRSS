import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime

host_path = 'http://ship-service.ship.svc.cluster.local:80/'


def w02_check_api_call(**kwargs):
    ti = kwargs['ti']
    api_path = '/api/risk-mappings/w02/task-status/'
    api_params = {
        'task_id': 'w02',
        'run_id': ti.run_id
    }
    transaction_id = ti.xcom_pull(key='w02_transaction_id')

    response = requests.get(host_path + api_path + transaction_id, params=api_params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        if data is not None and 'status' in data:
            if data['status'] == 'completed':
                return True
        else:
            return False
    else:
        print(f"Error: {response.status_code}")
        return False


def w02_api_call(execution_date, **kwargs):
    ti = kwargs['ti']
    formatted_date = execution_date.strftime('%Y%m%d')
    api_path = '/api/risk-mappings/w02'
    api_params = {
        'dates': formatted_date,
        'task_id': 'w02',
        'run_id': ti.run_id
    }

    response = requests.get(host_path + api_path, params=api_params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        ti.xcom_push(key='w02_transaction_id', value=data['transaction_id'])
    else:
        print('w02_api_call failed')

def w03_reanalysis_api_call(execution_date, **kwargs):
    api_path = '/api/risk-mappings/w03/reanalysis'
    api_params = {
        'max-hour': 48
    }

    requests.get(host_path + api_path, params=api_params)

def w06_api_call(execution_date, **kwargs):
    ti = kwargs['ti']
    formatted_date = execution_date.strftime('%Y%m%d')
    api_path = '/api/risk-mappings/w06'
    api_params = {
        'dates': formatted_date,
        'correction_map_id': ti.correction_map_id  
    }

    requests.get(host_path + api_path, params=api_params)


with DAG(
        'n_kcgsa_risk-mapping_d7_dag',
        default_args={
            'owner': 'airflow',
            'start_date': datetime(2024, 9, 28),
        },
        schedule_interval=timedelta(days=7),  # 매 시 실행
        catchup=False  # 과거 실행을 건너뛸지 여부 (False로 하면 과거 실행 건너뜀)
) as dag:
    w02_task = PythonOperator(
        task_id='w02',
        python_callable=w02_api_call
    )

    w02_wait_task = PythonSensor(
        task_id='w02_wait',
        python_callable=w02_check_api_call,
        mode='poke',  # 또는 'reschedule'로 설정 가능
        poke_interval=60,  # 60초 간격으로 DB를 체크
        timeout=3600  # 총 1시간 동안 상태를 확인
    )

    w03_reanalysis_task = PythonOperator(
        task_id='w03_reanalysis',
        python_callable=w03_reanalysis_api_call
    )
    
    w06_task = PythonOperator(
        task_id='w06',
        python_callable=w06_api_call
    )

    # api_task >> next_task
    w02_task >> w02_wait_task >> w06_task
