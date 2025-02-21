import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime
from airflow.utils.trigger_rule import TriggerRule

host_path = 'http://ship-service.ship.svc.cluster.local:80/'


def w01_ecmwf_check_api_call(**kwargs):
    ti = kwargs['ti']
    api_path = '/api/risk-mappings/w01/ecmwf/task-status/'
    api_params = {
        'task_id': 'w01_ecmwf',
        'run_id': ti.run_id
    }
    transaction_id = ti.xcom_pull(key='w01_ecmwf_transaction_id')

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


def w01_ecmwf_async_api_call(execution_date, **kwargs):
    ti = kwargs['ti']
    formatted_date = execution_date.strftime('%Y%m%d')
    api_path = '/api/risk-mappings/w01/ecmwf'
    api_params = {
        'dates': formatted_date,
        'task_id': 'w01_ecmwf',
        'run_id': ti.run_id
    }

    response = requests.get(host_path + api_path, params=api_params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        ti.xcom_push(key='w01_ecmwf_transaction_id', value=data['transaction_id'])
    else:
        print('w01_async_api_call failed')


def w01_noaa_check_api_call(**kwargs):
    ti = kwargs['ti']
    api_path = '/api/risk-mappings/w01/noaa/task-status/'
    api_params = {
        'task_id': 'w01_noaa',
        'run_id': ti.run_id
    }
    transaction_id = ti.xcom_pull(key='w01_noaa_transaction_id')

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


def w01_noaa_async_api_call(execution_date, **kwargs):
    ti = kwargs['ti']
    formatted_date = execution_date.strftime('%Y%m%d')
    api_path = '/api/risk-mappings/w01/noaa'
    api_params = {
        'dates': formatted_date,
        'task_id': 'w01_noaa',
        'run_id': ti.run_id
    }

    response = requests.get(host_path + api_path, params=api_params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        ti.xcom_push(key='w01_noaa_transaction_id', value=data['transaction_id'])
    else:
        print('w01_async_api_call failed')


def w01_kma_check_api_call(**kwargs):
    ti = kwargs['ti']
    api_path = '/api/risk-mappings/w01/kma/task-status/'
    api_params = {
        'task_id': 'w01_kma',
        'run_id': ti.run_id
    }
    transaction_id = ti.xcom_pull(key='w01_kma_transaction_id')

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


def w01_kma_async_api_call(execution_date, **kwargs):
    ti = kwargs['ti']
    formatted_date = execution_date.strftime('%Y%m%d')
    api_path = '/api/risk-mappings/w01/kma'
    api_params = {
        'dates': formatted_date,
        'task_id': 'w01_kma',
        'run_id': ti.run_id
    }

    response = requests.get(host_path + api_path, params=api_params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        ti.xcom_push(key='w01_kma_transaction_id', value=data['transaction_id'])
    else:
        print('w01_async_api_call failed')

def w03_ecmwf_api_call(execution_date, **kwargs):
    api_path = '/api/risk-mappings/w03/ecmwf'
    api_params = {
        'max-hour': 48
    }

    requests.get(host_path + api_path, params=api_params)


def w03_noaa_api_call(execution_date, **kwargs):
    formatted_date = execution_date.strftime("%Y%m%d%H") + "0000"
    api_path = '/api/risk-mappings/w03/noaa'
    api_params = {
        'date': formatted_date,
        'max-hour': 48
    }

    requests.get(host_path + api_path, params=api_params)


def w03_noaa_cyclone_api_call(execution_date, **kwargs):
    formatted_date = execution_date.strftime("%Y%m%d%H") + "0000"
    api_path = '/api/risk-mappings/w03/noaa/cyclone'
    api_params = {
        'date-time': formatted_date
    }

    requests.get(host_path + api_path, params=api_params)

def w03_kma_api_call(execution_date, **kwargs):
    formatted_date = execution_date.strftime("%Y%m%d%H") + "0000"
    api_path = '/api/risk-mappings/w03/kma'
    api_params = {
        'date': formatted_date,
        'max-hour': 48
    }

    requests.get(host_path + api_path, params=api_params)


def w07_api_call(execution_date, **kwargs):
    formatted_date = execution_date.strftime('%Y%m%d%H') + '0000'

    api_path = '/api/risk-mappings/w07'
    api_params = {
        'voyage_risk_map_id': formatted_date
    }

    requests.get(host_path + api_path, params=api_params)



def w08_api_call(execution_date, **kwargs):
    formatted_date = execution_date.strftime('%Y%m%d%H') + '0000'

    dag_run = kwargs['dag_run']
    task_instances = dag_run.get_task_instances()
    success_tasks = []
    for ti in task_instances:
        # 각 task instance의 id와 state를 확인
        task_id = ti.task_id
        state = ti.current_state()
        if state == "success":
            success_tasks.append(task_id)

    # 성공한 태스크 확인. 이걸 parameter로 넘김
    print(f"Success tasks: {', '.join(success_tasks)}")

    api_path = '/api/risk-mappings/w08'
    api_params = {
        'voyage_risk_map_id': formatted_date,
        'success_tasks': success_tasks
    }

    requests.get(host_path + api_path, params=api_params)


with DAG(
        'n_kcgsa_risk-mapping_h1_dag',
        default_args={
            'owner': 'airflow',
            'start_date': datetime(2024, 9, 28),
        },
        schedule_interval='@hourly',  # 매 시 실행
        catchup=False  # 과거 실행을 건너뛸지 여부 (False로 하면 과거 실행 건너뜀)
) as dag:
    w01_ecmwf_async_task = PythonOperator(
        task_id='w01_ecmwf',
        python_callable=w01_ecmwf_async_api_call
    )

    w01_ecmwf_wait_task = PythonSensor(
        task_id='w01_ecmwf_wait',
        python_callable=w01_ecmwf_check_api_call,
        mode='poke',  # 또는 'reschedule'로 설정 가능
        poke_interval=60,  # 60초 간격으로 DB를 체크
        timeout=3600  # 총 1시간 동안 상태를 확인
    )

    w01_noaa_async_task = PythonOperator(
        task_id='w01_noaa',
        python_callable=w01_noaa_async_api_call
    )

    w01_noaa_wait_task = PythonSensor(
        task_id='w01_noaa_wait',
        python_callable=w01_noaa_check_api_call,
        mode='poke',  # 또는 'reschedule'로 설정 가능
        poke_interval=60,  # 60초 간격으로 DB를 체크
        timeout=3600  # 총 1시간 동안 상태를 확인
    )

    w01_kma_async_task = PythonOperator(
        task_id='w01_kma',
        python_callable=w01_kma_async_api_call
    )

    w01_kma_wait_task = PythonSensor(
        task_id='w01_kma_wait',
        python_callable=w01_kma_check_api_call,
        mode='poke',  # 또는 'reschedule'로 설정 가능
        poke_interval=60,  # 60초 간격으로 DB를 체크
        timeout=3600  # 총 1시간 동안 상태를 확인
    )
    w03_ecmwf_task = PythonOperator(
        task_id='w03_ecmwf',
        python_callable=w03_ecmwf_api_call
    )

    w03_noaa_cyclone_task = PythonOperator(
        task_id='w03_noaa_cyclone',
        python_callable=w03_noaa_cyclone_api_call
    )

    w03_noaa_task = PythonOperator(
        task_id='w03_noaa',
        python_callable=w03_noaa_api_call
    )

    w03_kma_task = PythonOperator(
        task_id='w03_kma',
        python_callable=w03_kma_api_call
    )

    w07_task = PythonOperator(
        task_id='w07',
        python_callable=w07_api_call
    )

    w08_task = PythonOperator(
        task_id='w08',
        python_callable=w08_api_call,
        trigger_rule=TriggerRule.ALL_DONE  # 실패한 task가 있어도 w08은 실행됨
    )

    # api_task >> next_task
    w01_ecmwf_async_task >> w01_ecmwf_wait_task >> w03_ecmwf_task >> w07_task >> w08_task
    w01_noaa_async_task >> w01_noaa_wait_task >> w03_noaa_task >> w08_task
    w01_noaa_async_task >> w01_noaa_wait_task >> w03_noaa_cyclone_task >> w08_task
    w01_kma_async_task >> w01_kma_wait_task >> w03_kma_task >> w08_task
