# -*- coding: utf-8 -*-
"""
@Time          : 2025/01/09 00:00
@Author        : Hyunsun Lee
@File          : data_preprocessing.py
@Notice        : 
@Description   : Dataset definitions for vessel prediction tasks, including platform information.
@How to use    : Excuted by the main algorithm script.

@Modification :
    @Author    :
    @Time      :
    @Detail    :
"""

from pyproj import CRS, Transformer
from scipy.interpolate import PchipInterpolator
import numpy as np
import pandas as pd
import datetime as dt
import math


def epsg4326_to_3857(ais_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert coordinates from EPSG:4326 (WGS 84) to EPSG:3857 (Web Mercator).
    """
    projection_4326 = CRS("EPSG:4326")
    projection_3857 = CRS("EPSG:3857")
    transformer = Transformer.from_crs(projection_4326, projection_3857, always_xy=True)

    ais_data['lon'], ais_data['lat'] = zip(*ais_data.apply(
        lambda row: transformer.transform(row['lon'], row['lat']), axis=1
    ))
    return ais_data


class TestLoader:
    """
    Loader for testing ship trajectory data. Provides methods for interpolation and trajectory preparation.
    """

    def __init__(self):
        self.trajectory_dict = {}
        self.trajectory = []

    @staticmethod
    def datetime_to_datenum(dtime: dt.datetime) -> float:
        """
        Convert a datetime object to a MATLAB datenum format.
        """
        mdn = dtime + dt.timedelta(days=366)
        frac_seconds = (dtime - dt.datetime(dtime.year, dtime.month, dtime.day)).seconds / 86400.0
        frac_microseconds = dtime.microsecond / 86400.0 / 1e6
        return mdn.toordinal() + frac_seconds + frac_microseconds

    @staticmethod
    def filter_continuous_data(df: pd.DataFrame, max_gap_minutes: int = 120, min_duration_hours: int = 3) -> pd.DataFrame:
        """
        Filter continuous segments of data based on time gaps and minimum duration.
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60.0
        large_gaps = df[df['time_diff'] > max_gap_minutes].index

        start_idx = 0
        end_idx = len(df) - 1

        for gap_idx in large_gaps:
            if (df.loc[gap_idx - 1, 'datetime'] - df.loc[start_idx, 'datetime']).total_seconds() / 3600.0 >= min_duration_hours:
                end_idx = gap_idx - 1
                break
            start_idx = gap_idx

        if (df.loc[end_idx, 'datetime'] - df.loc[start_idx, 'datetime']).total_seconds() / 3600.0 < min_duration_hours:
            raise ValueError("No continuous segment found with the required duration and gap constraints.")

        return df.loc[start_idx:end_idx].reset_index(drop=True)

    def process_csv(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process the trajectory data by interpolating and preparing features.
        """
        df = self.data_interpolation(df)
        if df is None or df.empty:
            return np.array([])

        df['lon'] = df['lon'].round(5)
        df['lat'] = df['lat'].round(5)
        df['cog'] = df['cog'].round(3)
        df['sog'] = df['sog'].round(3)
        df['vx'] = df['vx'].round(3)
        df['vy'] = df['vy'].round(3)

        df['time_diff'] = pd.to_datetime(df['datetime'])
        df['delta_time'] = df['time_diff'].diff().dt.total_seconds() * 1000
        df['delta_lng'] = df['lon'].diff()
        df['delta_lat'] = df['lat'].diff()
        df['delta_cog'] = df['cog'].diff()
        df['delta_sog'] = df['sog'].diff()

        df['time_ms'] = df['datetime'].apply(lambda x: x.timestamp() * 1000)
        df.iloc[0, df.columns.get_loc('delta_time'):] = 0

        return df[['delta_time', 'delta_lng', 'delta_lat', 'vx', 'vy', 'time_ms', 'lon', 'lat']].iloc[1:].to_numpy()

    def data_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate the trajectory data for continuous time intervals.
        """
        if df.empty:
            print("Warning: DataFrame is empty!")
            return None

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.dropna(subset=['datetime', 'lon', 'lat', 'cog', 'sog']).drop_duplicates(subset=['datetime'])
        if df.empty:
            print("Warning: DataFrame is empty after cleaning!")
            return None

        df = df.sort_values('datetime')
        df['cog'] %= 360
        df['sog'] *= 0.51444444
        df['vx'] = df['sog'] * np.sin(np.radians(df['cog']))
        df['vy'] = df['sog'] * np.cos(np.radians(df['cog']))

        df.set_index('datetime', inplace=True)
        columns_to_interpolate = ['lon', 'lat', 'cog', 'sog', 'vx', 'vy']
        time_interval = dt.timedelta(minutes=10)

        start_time = df.index[0].replace(second=0, microsecond=0)
        end_time = df.index[-1]
        time_points = pd.date_range(start=start_time, end=end_time, freq=time_interval)

        x = list(map(self.datetime_to_datenum, df.index))
        unique_x, unique_indices = np.unique(x, return_index=True)
        df = df.iloc[unique_indices]
        interpolated_data = {column: PchipInterpolator(unique_x, df[column])(list(map(self.datetime_to_datenum, time_points))) for column in columns_to_interpolate}

        interpolated_df = pd.DataFrame(interpolated_data)
        interpolated_df.insert(0, 'datetime', time_points)
        return interpolated_df.drop_duplicates(subset=['datetime'])

    def load_test_trajectory(self, data: pd.DataFrame, seq_length: int):
        """
        Load trajectory data for testing, preparing sequences of the specified length.
        """
        data.columns = data.columns.str.lower()
        data = epsg4326_to_3857(data)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values(by='datetime')

        start_idx = 0
        for i in range(len(data) - 1):
            if data.iloc[i + 1]['datetime'] - data.iloc[i]['datetime'] > pd.Timedelta(hours=3):
                self._process_segment(data[start_idx:i], seq_length)
                start_idx = i + 1

        self._process_segment(data[start_idx:], seq_length)
        self.trajectory = np.array(self.trajectory)

    def _process_segment(self, segment: pd.DataFrame, seq_length: int):
        """
        Process a segment of trajectory data into sequences.
        """
        points = self.process_csv(segment)
        for i in range(len(points) - seq_length):
            self.trajectory.append(points[i:i + seq_length].astype('float32'))
