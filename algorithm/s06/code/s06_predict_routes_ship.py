import numpy as np
import pandas as pd
import json
import math
import time
from typing import Any, Dict, List, Tuple
from utils.cfg import Cfg
import logging
import os

#input_grd_file: str = 'W:/ship_ais/L1DVesselswithVelocity.json'


def coursePred(input_vessel_detection_file: str, time_interval: float = 10.0, time_end: float = 60.0) -> List[Dict[str, Any]]:
    """
    Predict the future course of vessels based on their initial positions and velocities.

    :param input_vessel_detection_file: Path to the JSON file containing L1D vessel data.
    :param time_interval: Time interval (in minutes) for each prediction step.
    :param time_end: End time (in minutes) for the prediction.
    :return: A list of dictionaries containing updated vessel data with predicted positions.
    """
    
    # Load json data: L1D vessel with velocity
    L1Dvesseldata = pd.read_csv(input_vessel_detection_file)
   
    time_range = np.linspace(time_interval, time_end, int(time_end / time_interval))

    # 각 시간에 대한 열 이름을 생성하여 데이터프레임에 추가
    for t in time_range:
        L1Dvesseldata[f'PredLon_{int(t)}'] = np.nan
        L1Dvesseldata[f'PredLat_{int(t)}'] = np.nan

    # 선박 데이터 순회하면서 예측 위치 계산
    for index, row in L1Dvesseldata.iterrows():
        lon = row['Lon']
        lat = row['Lat']
        heading = row['COG']
        velocity = row['SOG']

        # 각 시간에 대한 예측 위치 계산
        for t in time_range:
            new_lat, new_lon = predict_shipLocation(lat, lon, heading, velocity, t)
            L1Dvesseldata.at[index, f'PredLon_{int(t)}'] = new_lon
            L1Dvesseldata.at[index, f'PredLat_{int(t)}'] = new_lat
            
    return L1Dvesseldata


def predict_shipLocation(lat: float, lon: float, heading: float, velocity: float, time_mins: float) -> Tuple[float, float]:
    """
    Predict a ship's new position after a given time.

    :param lat: Initial latitude in degrees.
    :param lon: Initial longitude in degrees.
    :param heading: Heading in degrees (0-360).
    :param velocity: Velocity in m/s (assumed).
    :param time_mins: Time elapsed in minutes for which to predict the ship's location.
    :return: Tuple of (new_lat, new_lon) in degrees.
    """
    # Convert lat, lon, and heading from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    heading_rad = math.radians(heading)

    # Convert the time from minutes to seconds if needed (Currently, it seems time is used as is in m*s)
    # distance (meters) = velocity (m/s) * time (seconds)
    # But original code uses time_mins directly. So confirm the unit used. Here we'll assume time_mins in seconds:
    distance: float = velocity * time_mins

    # Earth radius in meters
    R = 6371000.0

    # Calculate the new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance / R)
        + math.cos(lat_rad) * math.sin(distance / R) * math.cos(heading_rad)
    )

    # Calculate the new longitude
    new_lon_rad = lon_rad + math.atan2(
        math.sin(heading_rad) * math.sin(distance / R) * math.cos(lat_rad),
        math.cos(distance / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )

    # Convert back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon


def get_args():
    """
    Parse command-line arguments for the vessel data file and prediction times.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_vessel_detection_file', 
        type=str,
        default='../data/input/2024-10-12-01-47-23_UMBRA-07_SICD_MM.txt',
        help='Path to the L1D vessel detection file w/ COG and SOG'
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='../data/output/',
        help='Path to the output file'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()

    args = get_args()
    print(args);print("start prediction vessel location!")

    L1Dvesseldata = coursePred(args.input_vessel_detection_file, Cfg.time_interval, Cfg.time_end)
    
    img_name = args.input_vessel_detection_file.split('/')[-1]
    L1Dvesseldata.to_csv(args.output_dir+img_name, index=False)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")