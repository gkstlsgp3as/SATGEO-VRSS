import numpy as np
import pandas as pd
import json
import math
import time
from typing import Any, Dict, List, Tuple
from cfg import Cfg

#input_grd_file: str = 'W:/ship_ais/L1DVesselswithVelocity.json'


def coursePred(input_grd_file: str, time_interval: float = 10.0, time_end: float = 60.0) -> List[Dict[str, Any]]:
    """
    Predict the future course of vessels based on their initial positions and velocities.

    :param input_grd_file: Path to the JSON file containing L1D vessel data.
    :param time_interval: Time interval (in minutes) for each prediction step.
    :param time_end: End time (in minutes) for the prediction.
    :return: A list of dictionaries containing updated vessel data with predicted positions.
    """
    startt: float = time.time()
    L1Dvesseldata: List[Dict[str, Any]] = []

    # Load json data: L1D vessel with velocity
    with open(input_grd_file, 'r') as file:
        for line in file:
            try:
                L1Dvesseldata.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    L1DvesseldataExp = L1Dvesseldata.copy()

    # Define reference time and predicted course for each vessel
    for num in range(len(L1Dvesseldata)):
        L1Dvesseldatatemp = L1Dvesseldata[num]

        lat: float = L1Dvesseldatatemp['Lat']
        lon: float = L1Dvesseldatatemp['Lon']
        heading: float = L1Dvesseldatatemp['COG']
        velocity: float = L1Dvesseldatatemp['SOG']

        # Create a range of times (in minutes) for predictions
        timeRange: np.ndarray = np.linspace(time_interval, time_end, round(time_end/time_interval))
        NewLat: List[float] = []
        NewLon: List[float] = []

        for num1 in range(len(timeRange)):
            new_lat, new_lon = predict_shipLocation(lat, lon, heading, velocity, timeRange[num1])
            NewLat.append(new_lat)
            NewLon.append(new_lon)

        L1Dvesseldatatemp['PredLon'] = NewLon
        L1Dvesseldatatemp['PredLat'] = NewLat
        L1Dvesseldatatemp['PredTime'] = timeRange

        L1DvesseldataExp[num] = L1Dvesseldatatemp

    print(time.time() - startt, ' [s]')

    return L1DvesseldataExp


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
        '--input_grd_file', 
        type=str,
        help='Path to the L1D vessel file'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    L1DvesseldataExp = coursePred(args.input_grd_file, Cfg.time_interval, Cfg.time_end)
