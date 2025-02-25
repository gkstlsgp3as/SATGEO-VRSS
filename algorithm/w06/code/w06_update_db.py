import os
import numpy as np
import datetime
import pandas as pd
import xarray as xr

def bias_db_update(file_bias, file_M2, sel_fct, df_era, df_obs, *args):
    """
    Update bias and M2 databases with a new forecast file.

    If the forecast (sel_fct) hasn't been processed yet, this function:
      1. Reads forecast and matching reanalysis data.
      2. Applies optimal interpolation to update reanalysis fields using observations.
      3. Updates bias (mean error) and variance (M2) using Welford's algorithm.
      4. Writes the updated databases back to their NetCDF files.

    Parameters:
        file_bias (str): Path to the bias database file.
        file_M2 (str): Path to the M2 database file.
        sel_fct (pd.Series): pandas Series of a forecast file info.
        df_era (pd.DataFrame): DataFrame of reanalysis file details.
        df_obs (pd.DataFrame): DataFrame of observation file details.

    Returns:
        0 on successful update;
        1 if the file is already processed, data is corrupted, or required reanalysis files are missing.

    Raises:
        ValueError: If the processed file lists in the bias and M2 databases are inconsistent.
    """
    # Constants for bias update (tuning parameters)
    c_T = 12        # Temporal correlation factor
    c_L = 0.5       # Spatial correlation length scale
    Rsq = 0.1       # Observation error variance factor
    
    # Variable names for forecast, reanalysis, and observation files
    vnames_fct = ["speed", "dir", "waveh", "waved"]
    vnames_era = ["speed", "dir", "waveh", "waved"]
    vnames_obs = ["SAR wind speed", "", "", ""]

    # Read the bias database's dlist to check if this forecast file has been used
    with xr.open_dataset(file_bias) as ds_bias:
      ls_bias = ds_bias['dlist'].values
    if sel_fct["name"] in ls_bias:
      return 1      # Skip update if file already in DB

    # Get forecast time from the forecast file 'sel_fct'
    file_fct = os.path.join(sel_fct["folder"], sel_fct["name"])
    with xr.open_dataset(file_fct) as ds_fct:
      time_fct = ds_fct['time'].values
    
    # Check if reanalysis files for all forecast times are available
    if all([any(is_samedate(t, a["time"]) for _,a in df_era.iterrows()) for t in time_fct]):
      # Read forecast
      with xr.open_dataset(file_fct) as ds_fct:
        fct = {'time': time_fct}
        try:    fct['lon'] = ds_fct['lon'].values
        except: fct['lon'] = ds_fct['longitude'].values
        try:    fct['lat'] = ds_fct['lat'].values
        except: fct['lat'] = ds_fct['latitude'].values
        for v in vnames_fct:
          fct[v] = ds_fct[v].values
      # Read (corresponding) reanalyses
      era = read_era(df_era, vnames_era, time_fct)
      
      # Check data integrity
      for v_fct, v_era in zip(vnames_fct,vnames_era):
        for i_t in range(len(time_fct)):
          if np.all( np.isnan(fct[v_fct][i_t,:,:]) ):
            print(f'forecast corrupted at {i_t}\'th. cancelled. ', end="")
            return 1
          if np.all( np.isnan(era[v_era][i_t,:,:]) ):
            print(f'reanalysis corrupted at {i_t}\'th. cancelled. ', end="")
            return 1
      
      # Update reanalyses using Optimal Interpolation
      for v_era, v_obs in zip(vnames_era,vnames_obs):
        if v_obs:
          # Select matching observation records for the forecast time range
          sel_obs = [a for _,a in df_obs.iterrows() if any(is_between(t, min(time_fct), max(time_fct)) for t in [a["time"]])]
          if sel_obs:
            print(f'OI {v_era}..', end="")
            era[v_era] = optimal_interp(era[v_era], era['lon'], era['lat'], time_fct, sel_obs, v_obs, file_M2, v_era, c_T, c_L, Rsq)

      # Read DB
      bias = read_db(file_bias)
      M2   = read_db(file_M2)
      
      # Check if dlist (list of processed files) are consistent between DBs
      if not np.array_equal(bias['dlist']['values'], M2['dlist']['values']):
        raise ValueError('db file lists inconsistent')
      
      # Update the bias DB using Welford's online algorithm for variance; https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
      bias['dlist']['values'] = np.append(bias['dlist']['values'], sel_fct["name"])
      M2['dlist']['values']   = np.append(M2['dlist']['values'],   sel_fct["name"])
      N = len(bias['dlist']['values']) + 1
      for v_fct, v_era in zip(vnames_fct,vnames_era):
        # TODO: Conditions can be further categorized based on each variable and its value.
        err_n = subtract(fct[v_fct], era[v_era], v_era)   # new error (fct - era)
        bias_o = bias[v_era]['values']                    # previous bias (mean error)
        bias_n = ( (N-1) * bias_o + err_n ) / N           # updated  bias (mean error)
        bias[v_era]['values'] = bias_n
        M2[v_era]['values'] += (err_n - bias_o) * (err_n - bias_n)
        # DEBUG: Check diverging
        check_diverging( M2[v_era]['values'], 1e7, "M2["+v_era+"]" )
      
      # Write DB
      write_db(file_bias,bias)
      write_db(file_M2,M2)
        
      return 0        # Update successful

    else:
      print('reanalyses not ready. cancelled. ', end="")
      return 1


def read_era(df_era, vnames, time_fct):
    """
    Read reanalysis data corresponding to forecast times.
    Initializes arrays for each variable and fills them using matching files.
    """
    file_era = os.path.join(df_era['folder'][0], df_era['name'][0])
    era = {'time': time_fct}
    with xr.open_dataset(file_era) as ds:
      era['lon'] = ds['lon'].values
      era['lat'] = ds['lat'].values
      # Initialize each variable's array with NaNs
      for v in vnames:
        era[v] = np.full((len(time_fct), ds[v].sizes['lat'], ds[v].sizes['lon']), np.nan)
    
    # Loop over forecast times and fill in the corresponding reanalysis data
    for i_t_fct in range(len(time_fct)):
      sel_era = [a for _,a in df_era.iterrows() if is_samedate(a["time"], time_fct[i_t_fct])]
      file_era = os.path.join(sel_era[-1]["folder"], sel_era[-1]["name"])
      with xr.open_dataset(file_era) as ds_era:
        try:
          time_era = ds_era['time'].values
        except KeyError:
          time_era = ds_era['valid_time'].values
        # Find the index in the reanalysis file that matches the forecast time
        i_t_era = [i for i, t in enumerate(time_era) if t == time_fct[i_t_fct]]
        for v in vnames:
          era[v][i_t_fct,:,:] = ds_era[v].values[i_t_era[0],:,:]
    
    return era


def read_db(file_db):
    """
    Read a NetCDF database file and store variables along with their values and dimensions.
    """
    db = {}
    with xr.open_dataset(file_db) as ds:
      coords = list(ds.coords)
      data_vars = list(ds.data_vars)
      db = {"coords": coords, "data_vars": data_vars}
      
      # Extract variables from both coords and data_vars
      for var in coords + data_vars:
        db[var] = {
          "values": ds[var].values,
          "dims": ds[var].dims
        }
    return db


def write_db(file_db,db):
    """
    Write a database (from read_db) to a NetCDF file.
    """
    ds = xr.Dataset()
    # Add coordinate variables using their original dims
    for coord in db.get("coords", []):
        data = db[coord]["values"]
        dims = db[coord]["dims"]
        ds = ds.assign_coords({coord: xr.DataArray(data, dims=dims)})
    # Add data variables using their original dims
    for var in db.get("data_vars", []):
        data = db[var]["values"]
        dims = db[var]["dims"]
        ds[var] = xr.DataArray(data, dims=dims)
    # Write the Dataset to a netCDF file (overwrite if exists)
    ds.to_netcdf(file_db)
    return 0

def optimal_interp(bkg, lon_bkg, lat_bkg, time_bkg, sel_obs, v_obs, file_M2, v_M2, c_T, c_L, Rsq):
    """
    Perform optimal interpolation on the background field using observation data.
    Returns the updated background field.
    """
    # Get observation file and time from the selected observation records
    file_obs = os.path.join(sel_obs[-1]["folder"], sel_obs[-1]["name"])
    time_obs = np.datetime64(sel_obs[-1]["time"])
    # Find index in the background time array matching the observation time   # TODO: interpolate bkg to compare with obs
    i_t_bkg = np.where(time_bkg == time_obs)[0]
    with xr.open_dataset(file_obs) as ds_obs:
      if not v_obs in list(ds_obs.data_vars):
        print(f'No {v_obs}',end='')
        return bkg
      obs = ds_obs[v_obs].values
      lon = ds_obs['lon'].values
      lat = ds_obs['lat'].values
    with xr.open_dataset(file_M2) as ds_M2:
      M2 = ds_M2[v_M2].values[i_t_bkg[0],:,:]
      N = ds_M2['dlist'].size
        
    bkg_n = bkg.copy()          # Initialize updated background with current values
    mask = ~np.isnan(obs)
    if mask.any():
      # Create meshgrids for observation and background grids
      Lon, Lat = np.meshgrid(lon, lat)
      Lon = Lon[mask]
      Lat = Lat[mask]
      Lon_bkg, Lat_bkg = np.meshgrid(lon_bkg, lat_bkg)
      
      # (flatten) background values and select observation values
      x = bkg[i_t_bkg[0],:,:].flatten()
      y = obs[mask]
      
      # Build the interpolation matrix (H) using bilinear weights
      H = np.zeros((len(y), len(x)))
      for iob in range(len(y)):
        w, _, i = weight_bil(lon_bkg, lat_bkg, Lon[iob], Lat[iob])
        H[iob, i.flatten()] = w.flatten()
      
      # Compute background error covariance (using Gaussian weights)
      Var = M2 / N if N > 0 else M2     # variance
      sigma = np.sqrt(Var)              # standard deviation
      B = np.outer(sigma.flatten(), sigma.flatten()) * weight_gaussian(Lon_bkg, Lat_bkg, c_L)
      B[B < 1e-9] = 0
      
      R = Rsq * np.eye(len(y))      # Observation error covariance matrix
      b = subtract(y, H @ x, v_M2)  # Innovation (obs - forecast)
      DEN = R + H @ B @ H.T         # Denomimator for the update equation
      x_up = B @ H.T @ np.linalg.inv(DEN) @ b     # Analysis increment
      x_up = x_up.reshape(bkg.shape[1], bkg.shape[2])
      
      # Apply a temporal weighting factor to the increment
      t_diff = (time_bkg - time_obs).astype('timedelta64[h]').astype(float)
      exp_factor = np.exp(-t_diff**2 / (2 * c_T**2))
      bkg_n = bkg + (x_up * exp_factor[:, np.newaxis, np.newaxis])
      # DEBUG: Check diverging
      check_diverging(x_up,1e3,"x_up")
    else:   # obs is NaN everywhere
      print('No Obs?..',end='')

    return bkg_n


def weight_bil(x, y, x_v, y_v):
    """
    Compute bilinear interpolation weights for a target point (x_v, y_v).
    
    Returns:
      weight: 2x2 array of weights.
      ij:     [i, j] indices for the lower-left grid cell.
      ilin:   Flattened indices for the 2x2 grid.
    """
    N = len(y)
    i = np.searchsorted(x, x_v)
    j = N - np.searchsorted(y[::-1], y_v) - 1
    if x[0]>x_v or x[-1]==x_v:
      i = i-1
    if y[-1]==y_v:
      j = N-2
    ij = [i, j]
    ilin = np.array([[ j*N + i, j*N + (i+1) ], [ (j+1)*N + i, (j+1)*N + (i+1) ]])
    x1 = x[i]
    x2 = x[i + 1]
    y1 = y[j]
    y2 = y[j + 1]
    weight = np.array([[(x2 - x_v) * (y2 - y_v), (x2 - x_v) * (y_v - y1)],
                       [(x_v - x1) * (y2 - y_v), (x_v - x1) * (y_v - y1)]]) / ((x2 - x1) * (y2 - y1))
    
    return weight, ij, ilin


def weight_gaussian(Lon, Lat, c_L):
    """
    Compute a Gaussian weight matrix for spatial covariance.
    """
    Lon_flat = Lon.flatten()
    Lat_flat = Lat.flatten()
    
    Lon_diff = Lon_flat[:, np.newaxis] - Lon_flat[np.newaxis, :]
    Lat_diff = Lat_flat[:, np.newaxis] - Lat_flat[np.newaxis, :]
    
    dsq = np.square(Lon_diff) + np.square(Lat_diff)
    rho = np.exp(-dsq / (2 * c_L**2))
    
    return rho


def is_samedate(dt1, dt2):
    """
    Check if two dates represent the same calendar day.
    """
    # if isinstance(dt1, datetime.datetime):
    #   dt1 = datetime.datetime(dt1.year, dt1.month, dt1.day).strftime('%Y-%m-%d')
    # elif isinstance(dt1, np.datetime64):
    #   dt1 = dt1.astype('datetime64[D]').astype(str)
    # if isinstance(dt2, datetime.datetime):
    #   dt2 = datetime.datetime(dt2.year, dt2.month, dt2.day).strftime('%Y-%m-%d')
    # elif isinstance(dt2, np.datetime64):
    #   dt2 = dt2.astype('datetime64[D]').astype(str)
    # return dt1 == dt2
    return np.datetime64(dt1, 'D') == np.datetime64(dt2, 'D')


def is_between(dt, start, end):
    """
    Check if a datetime is between start and end datetimes.
    """
    return start <= np.datetime64(dt) <= end


def subtract(x1, x2, v):
    """
    Compute the difference between x1 and x2.
    If v is 'dir' or 'waved', returns the minimal signed angular difference.
    Otherwise, returns the simple difference.
    """
    if v in ("dir", "waved"):
      return (x1 - x2 + 180) % 360 - 180
    else:
      return x1 - x2


def check_diverging(val, tol=1e16, title="it"):
    """
    Check if any value in the array 'val' exceeds the tolerance 'tol'.
    """
    i_large = np.argwhere(abs(val) > tol)
    if i_large.size > 0:
      print(f'Caution: {title} is too large. Check indices below.')
      print(i_large)


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


def bias_update_auto(data_root):
    """
    Automatically update the bias and M2 databases using forecast files.
    Processes only forecast files not yet used in the databases.
    """
    df_era, df_obs, df_fct = list_files(data_root)
    t_lap = []
    file_bias = "db_bias.nc"
    file_M2   = "db_M2.nc"
    
    # Read the list of forecast files already in the bias DB
    with xr.open_dataset(file_bias) as ds_bias:
      info_db_list = ds_bias['dlist'].values
    
    # Process only forecast files not already in the DB
    if len(df_fct) > 0:
      df_fct = df_fct[~df_fct["name"].isin(info_db_list)]
      for i_f, row in df_fct.iterrows():
        print(f'  db_update: {row["name"]}.. ', end="",flush=True)
        lap = datetime.datetime.now()
        bias_db_update(file_bias, file_M2, row, df_era, df_obs)
        t_lap.append( (datetime.datetime.now() - lap).total_seconds() )
        print(f'{t_lap[-1]:.3f} s')

    return 0

    
# Main function execution
data_root = "tmpdata"
bias_update_auto(data_root)
