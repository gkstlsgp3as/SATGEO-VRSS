import boto3
import os
import zipfile
from typing import List
import datetime
import shutil

from app.config.settings import settings
from sqlalchemy.orm import Session
from uvicorn.config import logger

from app.models.noaa_collect_hist import NoaaCollectHist
from app.service import noaa_collect_hist_service

# AWS S3에 접속하기 위한 설정
s3 = boto3.client('s3',
                  aws_access_key_id=settings.AWS_S3_ACCESS_KEY,
                  aws_secret_access_key=settings.AWS_S3_ACCESS_SECRET,
                  region_name=settings.AWS_S3_REGION)

# 다운로드할 파일이 위치한 S3 버킷과 파일 경로
bucket_name = settings.AWS_S3_BUCKET_NAME

# 다운로드할 로컬 경로
local_file_path = settings.W01_NOAA_OUTPUT_PATH + '/'


def download_files(target_date: str) -> None:
    logger.info(f'download noaa starts : {target_date}')

    for dir_info in get_dir_info(target_date):
        try:
            # 디렉터리가 존재하지 않으면 생성
            directory = os.path.dirname(dir_info['file_full_path'])
            if not os.path.exists(directory):
                os.makedirs(directory)

            # S3에서 파일 다운로드
            s3.download_file(bucket_name, dir_info['file_key'], dir_info['file_full_path'])
            print(f"File downloaded successfully to {dir_info['file_full_path']}")

            # ZIP 파일 열기
            with zipfile.ZipFile(dir_info['file_full_path'], 'r') as zip_ref:
                # 모든 파일을 지정한 폴더에 추출
                zip_ref.extractall(dir_info['extract_path'])
                print(f"압축 해제 완료: {dir_info['extract_path']}")

        except Exception as e:
            print(f"Error downloading file: {e}")


def remove_files(target_date: str) -> None:
    logger.info(f'remove noaa starts : {target_date}')

    for dir_info in get_dir_info(target_date):
        try:
            directory = os.path.dirname(dir_info['file_full_path'])
            # 폴더 내 모든 파일과 하위 폴더 삭제
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        # 디렉토리인 경우 재귀적으로 삭제
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        # 파일인 경우 삭제
                        else:
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            else:
                print(f"The folder {directory} does not exist.")

        except Exception as e:
            print(f"Error removing files: {e}")


def get_dir_info(target_date: str):
    dir_info = []
    formatted_date = target_date[:4] + '/' + target_date[4:6] + '/' + target_date[6:] + '/'
    file_path = local_file_path + target_date + '/'

    gfs_file_key = 'OceanWeather/gfs/' + formatted_date + '00/Pressure/gfs.t00z.pgrb2.0p25.f000.grib2_press.zip'
    dir_info.append({
        'file_key': gfs_file_key,
        'file_full_path': file_path + gfs_file_key.split('/')[-1],
        'extract_path': file_path + 'gfs'
    })

    hycom_file_key = 'OceanWeather/hycom/' + formatted_date + '00/Current/hycom_glb_sfc_u_' + target_date + '00_t000_current.zip'
    dir_info.append({
        'file_key': hycom_file_key,
        'file_full_path': file_path + hycom_file_key.split('/')[-1],
        'extract_path': file_path + 'hycom'
    })

    gfswave_file_key = 'OceanWeather/nww3/' + formatted_date + '00/Wave/gfswave.t00z.global.0p25.f000_wave.zip'
    dir_info.append({
        'file_key': gfswave_file_key,
        'file_full_path': file_path + gfswave_file_key.split('/')[-1],
        'extract_path': file_path + 'gfswave'
    })

    gfswind_file_key = 'OceanWeather/nww3/' + formatted_date + '00/Wave/gfswind.t00z.global.0p25.f000_wind.zip'
    dir_info.append({
        'file_key': gfswind_file_key,
        'file_full_path': file_path + gfswind_file_key.split('/')[-1],
        'extract_path': file_path + 'gfswind'
    })

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='OceanWeather/noaa_cyclone/' + formatted_date)
    noaa_cyclone_keys = [obj['Key'] for obj in response.get('Contents', [])]
    for key in noaa_cyclone_keys:
        dir_info.append({
            'file_key': key,
            'file_full_path': file_path + key.split('/')[-1],
            'extract_path': file_path + 'noaa_cyclone'
        })

    return dir_info


def lists_by_dates(db: Session, collect_history: NoaaCollectHist, dates: List[str]):
    for date in dates:
        download_files(date)

        collect_history.status = "completed"
        collect_history.updated_at = datetime.datetime.now()
        logger.info(f"====== updated: {collect_history.updated_at} ======")
        noaa_collect_hist_service.update_noaa_collect_history(db, collect_history)


def get_task_status(db: Session, task_id: str, run_id: str, transaction_id: str):
    return noaa_collect_hist_service.get_noaa_collect_history(db, task_id, run_id, transaction_id)

# if __name__ == "__main__":
    # args = parse_args()
    # main(args.date)
    # download_files('20240904')
