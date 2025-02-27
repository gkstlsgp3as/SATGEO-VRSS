import glob
import numpy as np
import xarray as xr
from sqlalchemy.orm import Session
import time
import argparse 
import logging
from VRSS.config.settings import settings

def init_db(file_db,path_ref):
    """
    Initialize a NetCDF database file using a reference file as a template.
    
    This function creates a Dataset with the same spatial (lat, lon) and temporal (time) dimensions
    as the first file matching the given reference path (glob pattern).
    All data variables are initialized with zeros, and an empty 'dlist' is added to track processed files.
    
    Parameters:
      file_db (str): The path to the database file to be created or overwritten.
      path_ref (str): A glob pattern to locate reference NetCDF files. The first matching
                      file is used as a template for dimensions and variables.
    """
    with xr.open_dataset(glob.glob(path_ref)[0]) as ds_sample:
      # - Define the required database information
      ds_db = xr.Dataset(coords={
                            "time": (("time",), np.arange(ds_sample['time'].size)),
                            "lat": ds_sample["lat"],
                            "lon": ds_sample["lon"],
                            }
                        )
      for v in list(ds_sample.data_vars.keys()):
        ds_db[v] = xr.DataArray(np.zeros(ds_sample[v].shape), dims=ds_sample[v].dims)
      ds_db['dlist'] = xr.DataArray(np.array([], dtype=str), dims=("list",))

    # - Overwrite existing database files
    ds_db.to_netcdf(file_db)


def process(db: Session, correction_map_id: str):
  # - main function
  
  ## get the input nc files from db 
  work_dir = settings.w06_WORK_DIR
  
  init_db(work_dir+"db_bias.nc","tmpdata/ECMWF_forecast_ready/*.nc")
  init_db(work_dir+"db_M2.nc",  "tmpdata/ECMWF_forecast_ready/*.nc")
  print("Database initialized successfully.")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_ecmwf_forecast_files",
      type=str,
      default="../data/input/ECMWF_forecast_ready/*.nc",
      required=True,
      help="path to ecmwf forecast nc files.",
  )
  parser.add_argument(
      "--output_bias_file",
      type=str,
      default="../data/output/db_bias.nc",
      required=False,
      help="path to output bias file"
  )
  parser.add_argument(
      "--output_m2_file",
      type=str,
      default="../data/output/db_M2.nc",
      required=False,
      help="path to output M2 file"
  )
  
  args = parser.parse_args()

  return args
  

if __name__ == "__main__":
  start_time = time.time()

  args = get_args()

  init_db(args.output_bias_file, args.input_ecmwf_forecast_files)
  init_db(args.output_m2_file,  args.input_ecmwf_forecast_files)
  
  processed_time = time.time() - start_time
  logging.info(f"{processed_time:.2f} seconds")
