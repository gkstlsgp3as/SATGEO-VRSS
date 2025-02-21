# -*- coding: utf-8 -*-
'''
@Time          : 
@Author        : 
@File          : 
@Noice         : 
@Description   : 
@How to use    : 

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

from utils.cfg import Cfg
import logging
import time

from sqlalchemy.orm import Session

from app.config.settings import settings
#from app.models.{모델_py_파일} import {모델_클래스}
from app.service.{서비스_py_파일} import {서비스_함수}

def sub_algorithm():
    ## 코드

def algorithm(input_dir: type, output_tif_file: type, input_meta_file: type):
    
    ## TODO
    # 원래 코드

    return results

    
def process(db: Session, parameter: str):
    
    start_time = time.time()
    
    input_dir = settings.SAMPLE_INPUT_PATH
    output_tif_file = settings.SAMPLE_OUTPUT_PATH
    input_meta_file = settings.SAMPLE_META_FILE
    
    args = Cfg.args
    
    results = algorithm(input_dir, output_tif_file, input_meta_file, args)
    
    # 서비스 함수 호출
    bulk_upsert_data_hist(db, results)
    
    # Calculate and log the processing time
    processed_time = time.time() - start_time
    logging.info(f"Processed SAR image classification in {processed_time:.2f} seconds")

''' # 인자 변경을 위한 참고용
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ALGORITHM_NAME)
    parser.add_argument('-i', "--input_dir", type=str, required=True, default="/platform/data/inputs/, help="Path to input images")
    parser.add_argument('-o', "--output_dir", type=str, required=True, default="/platform/data/outputs/predictions.csv", help="Path to save output CSV")
    parser.add_argument('-m', '--meta_file', type=str, required=True, help="Path to meta information file")

    args = parser.parse_args()
    
    algorithm(**vars(args))
'''
    
    
