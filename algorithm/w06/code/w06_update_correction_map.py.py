import os
import time
import numpy as np
import datetime
import pandas as pd
import xarray as xr

def bias_correction(sel_fc, dir_fc_cor, fname_db, vnames=None, attempts=3):
    """
    Apply bias correction to a forecast file.
    
    Parameters:
      sel_fc (dict):    Contains folder and name for the forecast file.
      dir_fc_cor (str): Directory where corrected forecast files will be saved.
      fname_db (str):   Path to the bias database NetCDF file.
      vnames (list, optional):  List of variable names to correct.
      attempts (int, optional): Number of attempts if file I/O fails.
    """
    for i in range(attempts):
      try:
        # Initialize dictionaries for bias, forecast, and corrected forecast data.
        bias, fc, fc_cor = {}, {}, {}
        # Open bias database and extract bias values for each variable (except 'dlist').
        with xr.open_dataset(fname_db) as dst:
          if vnames is None:
            vnames = [v for v in list(dst.data_vars) if v not in ('dlist',)]
          for v in vnames:
            bias[v] = dst[v].values

        # Warn if the bias values are all zero.
        for v in vnames:
          if np.all(bias[v]==0):
            print(f'DB not ready for {v}. ',end="")

        # Construct file paths for the forecast file and its corrected version.
        file_fc = os.path.join(sel_fc["folder"], sel_fc["name"])
        file_fc_cor = os.path.join(dir_fc_cor, sel_fc["name"])
        with xr.open_dataset(file_fc, decode_times=False) as ds_fc:
          for v in vnames:
            fc[v] = ds_fc[v].values
            # Correct forecasts            # TODO: Refine correction condition
            corred = fc[v] + bias[v]
            # Post-process corrections for specific conditions.
            if v in ("speed","waveh"):
              # Ensure magnitude values are not negative.
              corred[corred < 0] = 0
            elif v in ("dir", "waved"):
              # Adjust angle values to be 0-360 degrees.
              corred = corred % 360
            # Save corrected
            fc_cor[v] = corred
    
          # Create a new dataset and replace variables with corrected data
          ds_fc_cor = ds_fc.copy(deep=True)
          for v in vnames:
            ds_fc_cor[v] = (ds_fc[v].dims, fc_cor[v])
          # Save the corrected dataset as a NetCDF file
          ds_fc_cor.to_netcdf(file_fc_cor, format='NETCDF4')
          return 0      # Correction successful
      except:
        if i < attempts - 1:
          time.sleep(0.1)     # Wait 0.1 seconds before retrying.
        else:
          print(f'Correction failed after {attempts} attempts. ',end='')
          return 1      # Failed


def list_files(dir_data):
    """
    List NetCDF files in specific subdirectories and return DataFrames.
    
    Returns:
      df_era: DataFrame for reanalysis files.
      df_obs: DataFrame for observation files.
      df_fct: DataFrame for forecast files.
    """
    d = os.path.join(dir_data, "ECMWF_reanalysis_ready")
    df_era = pd.DataFrame(columns=["folder", "name", "time"])
    for f in os.listdir(d):
      if f.endswith('.nc'):
        t = datetime.datetime.strptime(f.replace('z.nc', ''), '%Y_%m_%d_%H')
        df_era.loc[len(df_era)] = [d, f, t]
    
    d = os.path.join(dir_data, "SAR_corr")
    df_obs = pd.DataFrame(columns=["folder", "name", "time"])
    for f in os.listdir(d):
      if f.endswith('_coarse.nc'):
        t = datetime.datetime.strptime(f.replace('z_coarse.nc', ''), '%Y_%m_%d_%H')
        df_obs.loc[len(df_obs)] = [d, f, t]
    
    d = os.path.join(dir_data, "ECMWF_forecast_ready")
    df_fct = pd.DataFrame(columns=["folder", "name", "time"])
    for f in os.listdir(d):
      if f.endswith('.nc'):
        t = datetime.datetime.strptime(f.replace('z.nc', ''), '%Y_%m_%d_%H')
        df_fct.loc[len(df_fct)] = [d, f, t]
    
    return df_era, df_obs, df_fct


def bias_correction_auto(data_root):
    """
    Automatically perform bias correction on forecast files.
    
    - Lists forecast files.
    - Discards those that have already been corrected.
    - Applies bias correction to each remaining forecast file.
    
    Returns:
      DataFrame of forecast files that were processed.
    """

    dir_fc_cor = os.path.join(data_root, "ECMWF_forecast_corr")
    df_era, df_obs, df_fct = list_files(data_root)
    
    if len(df_fct) > 0:
      # Discard previously corrected items
      if not os.path.exists(dir_fc_cor): os.mkdir(dir_fc_cor)
      list_cor = [f for f in os.listdir(dir_fc_cor) if f.endswith('.nc')]
      df_fct = df_fct[~df_fct["name"].isin(list_cor)]
      # Iterate over each forecast file for correction
      t_lap = []
      for i_f, row in df_fct.iterrows():
        print(f'  db correct: {row["name"]}.. ', end="",flush=True)
        lap = datetime.datetime.now()
        bias_correction(row, dir_fc_cor, "db_bias.nc")
        t_lap.append((datetime.datetime.now() - lap).total_seconds())
        print(f'{t_lap[-1]:.3f} s')

    return df_fct


# Main function execution
data_root = "tmpdata"
df_new = bias_correction_auto(data_root)
