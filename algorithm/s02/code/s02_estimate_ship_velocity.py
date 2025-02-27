from models.Umbra_velocity_estimation import *
import pandas as pd
import json
import argparse
from tifffile import imwrite
from pathlib import Path
from utils.cfg import Cfg

from sqlalchemy.orm import Session
import time
import logging


def process(db:Session, satellite_sar_image_id:str): 
    from app.service import sar_ship_unidentification_service 
    from app.config.settings import settings
    
    start_time = time.time()
    
    input_slc_dir = settings.S01_INPUT_PATH # S01_INPUT_PATH로 
    output_chip_dir = settings.S02_CHIP_PATH
    input_bbox_file = settings.S01_OUTPUT_PATH # 미식별에 대한거면 S03 output이어야 하나, 현재 S03이 Sentinel1에 대해서만 구현되어 있음.
    input_meta_file = settings.S02_META_FILE
    input_bbox_file = [f for f in os.listdir(input_bbox_file) if f.endswith('txt')][0]  # need considering multiple files? 
    
    vessel_db_data = sar_ship_unidentification_service.get_sar_ship_unidentification(db, satellite_sar_image_id)
    vessel_db_dict = [record.__dict__ for record in vessel_db_data]
    
    Velocticy_est_UMBRA_instance = Velocticy_est_UMBRA(input_slc_dir, start_Azimuthvel=Cfg.start_az, end_Azimuthvel=Cfg.end_az, spacing=Cfg.spacing, input_bbox_file=input_bbox_file)
    data_SLC_2DBox, SlantRangeBox, f_azimuth, f_range = Velocticy_est_UMBRA_instance.patchmaker() ## extracting RoI in SLC data
    nested_array = Velocticy_est_UMBRA_instance.velocity_estimation(data_SLC_2DBox, SlantRangeBox, f_azimuth, f_range) # adding velocity phase to SLC
    
    entropy_nes_array = Velocticy_est_UMBRA_instance.calculate_entropy(nested_array) # calculating entropy of each target chip
    vc_azimuth = Velocticy_est_UMBRA_instance.calculate_velocity(entropy_nes_array) # find minumum entropy for each target chip
    refocused_targetchip = Velocticy_est_UMBRA_instance.extract_rftarget(nested_array, entropy_nes_array)# refocusing target chip has refocused target array
    
    # extract heading angle from metadata
    ## from extract_parameter import Umbra
    ## meta_params = Umbra(input_slc_dir, input_meta_file)     # 위성 영상에 따라 Umbra: nitf, ICEYE: meta, K5: meta
    heading_angle = 10 # meta_params[5] # 확인 필
    target_velocity = Velocticy_est_UMBRA_instance.COGEstimation_vessel(refocused_targetchip, vc_azimuth, heading_angle)
    
    # make total array from bbox_array and azimuth velocity
    total_array = Velocticy_est_UMBRA_instance.bbox_array;total_array = total_array.astype(int) 
    total_array = np.column_stack((total_array, vc_azimuth))

    #check the output_chip_dir and save the refocused target chips to the output_chip_dir
    output_chip_dir = Path(output_chip_dir)  # increment run
    output_chip_dir.mkdir(parents=True, exist_ok=True)

    [imwrite(output_chip_dir / f'image_{i}.tif', np.abs(img)) for i, img in enumerate(refocused_targetchip)]

    # Loop through each row of the array
    vessel_db_dict['prediction_cog'] = target_velocity[:, 4]
    vessel_db_dict['prediction_sog'] = target_velocity[:, 5]  # 확인 필요

    # Convert the list of dictionaries to a DataFrame for bulk database update
    df = pd.DataFrame(vessel_db_dict)

    # Update the database with the new ship classification predictions
    sar_ship_unidentification_service.bulk_upsert_sar_ship_unidentification_velocity(db, df)
    
    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_slc_dir', 
        type=str, 
        default='../data/input/', 
        help='the file path that SICD and METADATA are stored'
    )
    parser.add_argument(
        '--input_bbox_file', 
        type=str, 
        default="../data/input/2024-10-12-01-47-23_UMBRA-07_SICD_MM.txt", 
        help='bounding box file path'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../data/output/', 
        help='path to save outputs'
    )
    parser.add_argument(
        '-C', "--output_chip_dir", 
        type=str, 
        default='../data/output/chips/', 
        help="Path to output chip images"
    )

    # Parse arguments
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    start_time = time.time()
    
    args = get_args()
    print(args);print("start estimating velocity!")

    Velocticy_est_UMBRA_instance = Velocticy_est_UMBRA(args.input_slc_dir, start_Azimuthvel=Cfg.start_az, end_Azimuthvel=Cfg.end_az, spacing=Cfg.spacing, input_bbox_file=args.input_bbox_file)
    data_SLC_2DBox, SlantRangeBox, f_azimuth, f_range = Velocticy_est_UMBRA_instance.patchmaker() ## extracting RoI in SLC data
    nested_array = Velocticy_est_UMBRA_instance.velocity_estimation(data_SLC_2DBox, SlantRangeBox, f_azimuth, f_range) # adding velocity phase to SLC
    
    entropy_nes_array = Velocticy_est_UMBRA_instance.calculate_entropy(nested_array) # calculating entropy of each target chip
    vc_azimuth = Velocticy_est_UMBRA_instance.calculate_velocity(entropy_nes_array) # find minumum entropy for each target chip
    refocused_targetchip = Velocticy_est_UMBRA_instance.extract_rftarget(nested_array, entropy_nes_array)# refocusing target chip has refocused target array
    
    heading_angle = 10 # meta_params[5] # 확인 필
    target_velocity = Velocticy_est_UMBRA_instance.COGEstimation_vessel(refocused_targetchip, vc_azimuth, heading_angle)
    
    # make total array from bbox_array and azimuth velocity
    total_array = Velocticy_est_UMBRA_instance.bbox_array;total_array = total_array.astype(int) 
    total_array = np.column_stack((total_array, vc_azimuth))

    #check the output_chip_dir and save the refocused target chips to the output_chip_dir
    output_chip_dir = Path(args.output_chip_dir)  # increment run
    output_chip_dir.mkdir(parents=True, exist_ok=True)

    [imwrite(output_chip_dir / f'image_{i}.tif', np.abs(img)) for i, img in enumerate(refocused_targetchip)]
    
    #search detected_label
    img_name = [f for f in os.listdir(args.input_slc_dir) if f.endswith('.txt')][0] # 파일 하나 가정 
    df = pd.read_csv(args.input_bbox_file); df['COG'] = target_velocity[:,4]; df['SOG'] = target_velocity[:,5]
    df.to_csv(args.output_dir+img_name, index=False)
    
    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")
