# Helper functions for loading CAFE data

import glob

import warnings

import xarray as xr

from .utils import (
    round_to_start_of_month, 
    convert_time_to_lead, 
    truncate_latitudes,
)


CAFEf6_dir = "/g/data/xv83/dcfp/CAFE-f6/"
CAFEf5_dir = "/g/data/xv83/dcfp/CAFE-f5/"
CAFE60_dir = "/g/data/xv83/dcfp/CAFE60v1/"
CAFEhist_path = "/g/data/xv83/users/ds0092/data/CAFE/historical/WIP"


def rescale_variables(ds):
    """ Rescale CAFE variables """
    
    SCALE_BY = {
        # Atmosphere
        "t_ref": 1, 
        "precip": 86400,
        
        # Ocean
        "sst": 1,
    }
    
    for v in ds.data_vars:
        if v in SCALE_BY.keys():
            ds[v] = ds[v] * SCALE_BY[v]
        else:
            warnings.warn(f"No scale factor available for variable {v}. Using 1")
    return ds
    

def preprocess_forecast(ds):
    """ Preprocessing steps for CAFE forecast data """
    ds = convert_time_to_lead(ds)
    # Truncate lats so that forecasts run on different systems can be stacked
    if "lat" in ds.dims:
        ds = truncate_latitudes(ds)
    return ds


def open_f6(realm, variables, preprocess=None):
    """ Open CAFE-f6 forecast data """
    files = sorted(
        glob.glob(f"{CAFEf6_dir}/c5-d60-pX-f6-????1101/{realm}.zarr.zip")
    ) # Skip May starts
    ds = xr.open_mfdataset(
        files,
        compat="override",
        preprocess=lambda x: preprocess(preprocess_forecast(x)),
        engine="zarr",
        coords="minimal",
        parallel=True,
    )[variables]
    
    ds = round_to_start_of_month(ds, dim=["init", "time"])
    ds = rescale_variables(ds)        
    return ds


def open_f5(realm, variables, preprocess=None):
    """Open CAFE-f5 forecast data"""
    ds = xr.open_dataset(
        f"{CAFEf5_dir}/NOV/{realm}.zarr.zip", engine="zarr", chunks={}
    )[variables]  # Skip May starts
    ds = ds.rename({"init_date": "init", "lead_time": "lead"})
    
    # Append 2020 forecast from CAFE-f6
    ds_2020 = xr.open_dataset(
        f"{CAFEf6_dir}/c5-d60-pX-f6-20201101/{realm}.zarr.zip", engine="zarr", chunks={}
    )[variables]
    ds_2020 = preprocess_forecast(ds_2020)
    
    ds = ds.assign_coords({"time": ds["time"].compute()}) # Required for concat below
    ds = xr.concat([ds, ds_2020.isel(ensemble=range(10))], dim="init")
    
    if preprocess is not None:
        ds = preprocess(ds)
    ds = round_to_start_of_month(ds, dim=["init", "time"])
    ds = rescale_variables(ds)
    return ds


def open_d60(realm, variables, preprocess=None):
    """ Open CAFE60v1 data """
    ds = xr.open_dataset(f"{CAFE60_dir}/{realm}.zarr.zip", engine="zarr", chunks={})[
        variables
    ]
    
    ds = truncate_latitudes(ds)
    if preprocess is not None:
        ds = preprocess(ds)
    ds = round_to_start_of_month(ds, dim="time")
    ds = rescale_variables(ds)
    return ds


def open_h0(realm, variables, preprocess=None):
    """Open CAFE historical run data"""

    hist = xr.open_dataset(
        f"{CAFEhist_path}/c5-d60-pX-hist-19601101/ZARR/{realm}.zarr.zip",
        engine="zarr",
        chunks={},
    )[variables]
    
    ctrl = xr.open_dataset(
        f"{CAFEhist_path}/c5-d60-pX-ctrl-19601101/ZARR/{realm}.zarr.zip",
        engine="zarr",
        chunks={},
    )[variables]
    
    hist = truncate_latitudes(hist)
    ctrl = truncate_latitudes(ctrl)
    if preprocess is not None:
        hist = preprocess(hist)
        ctrl = preprocess(ctrl)
    hist = round_to_start_of_month(hist, dim="time")
    ctrl = round_to_start_of_month(ctrl, dim="time")
    hist = rescale_variables(hist)
    ctrl = rescale_variables(ctrl)
    
    drift = (
        ctrl.mean("ensemble")
        .groupby("time.month")
        .map(lambda x: x - x.mean(["time"]))
    )
    return hist - drift
