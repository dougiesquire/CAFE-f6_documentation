import os
import sys
import tempfile
from pathlib import Path

from functools import reduce, partial

import yaml

import numpy as np
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_config(name):
    """
    Load a config .yml file for a specified dataset

    Parameters
    ----------
    name : str
        The path to the config file to load
    """
    with open(name, "r") as reader:
        return yaml.load(reader, Loader=yaml.SafeLoader)


def composite_function(function_dict):
    """
    Return a composite function of all functions and kwargs specified in a
    provided dictionary

    Parameters
    ----------
    function_dict : dict
        Dictionary with functions in this module to composite as keys and
        kwargs as values
    """

    def composite(*funcs):
        def compose(f, g):
            return lambda x: g(f(x))

        return reduce(compose, funcs, lambda x: x)

    funcs = []
    for fn in function_dict.keys():
        kws = function_dict[fn]
        kws = {} if kws is None else kws
        funcs.append(
            partial(getattr(sys.modules[__name__], fn), **kws)
        )  # getattr(utils, fn)

    return composite(*funcs)


def get_lon_lat_box(ds, box, lon_dim="lon", lat_dim="lat"):
    """
    Return a region specified by a range of longitudes and latitudes.

    Assumes data are on a regular grid.

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        The data to subset
    box : iterable
        Iterable with the following elements in this order:
        [lon_lower, lon_upper, lat_lower, lat_upper]
    lon_dim : str, optional
        The name of the longitude dimension
    lat_dim : str, optional
        The name of the latitude dimension
    """
    return ds.sel({lon_dim: slice(box[0], box[1]), lat_dim: slice(box[2], box[3])})


def get_lon_lat_average(ds, box, lon_dim="lon", lat_dim="lat"):
    """
    Return the average over a region specified by a range of longitudes and latitudes.

    Assumes data are on a regular grid.

    Parameters
    ----------
    ds : xarray Dataset or DataArray
        The data to subset and average. Assumed to include an "area" Variable
    box : iterable
        Iterable with the following elements in this order:
        [lon_lower, lon_upper, lat_lower, lat_upper]
    lon_dim : str, optional
        The name of the longitude dimension
    lat_dim : str, optional
        The name of the latitude dimension
    """
    return (
        get_lon_lat_box(ds, box, lon_dim, lat_dim)
        .weighted(ds["area"])
        .mean([lon_dim, lat_dim])
    )


def calculate_amv(sst_anom, sst_name="sst"):
    """
    Calculate the Atlantic Multi-decadal Variability (AMV)--also known as the Atlantic
    Multi-decadal Oscillation (AMO)--according to Trenberth and Shea (2006). The AMV
    is calculated as the spatial average of SST anomalies over the North Atlantic
    (Equator–60∘ N and 80–0∘ W) minus the spatial average of SST anomalies averaged from
    60∘ S to 60∘ N.

    Longitude is assumed to range from 0-360 deg.

    Note typically the SST anomalies are smoothed in time using a 10-year moving average
    (Goldenberg et al., 2001; Enfield et al., 2001), a low-pass filter (Trenberth and Shea
    2006) or a 4-year temporal average (Bilbao at al., 2021).

    Parameters
    ----------
    sst_anom : xarray Dataset
        Array of sst anomalies
    sst_name : str, optional
        The name of the sst variable in sst_anom
    """

    north_atlantic_box = [280, 360, 0, 60]
    global_box = [0, 360, -60, 60]

    amv = get_lon_lat_average(sst_anom, north_atlantic_box) - get_lon_lat_average(
        sst_anom, global_box
    )
    amv = amv.rename({sst_name: "amv"})
    amv["amv"].attrs = dict(
        long_name="Atlantic multi-decadal variability", units="degC"
    )
    return amv


def calculate_ipo(sst_anom, sst_name="sst"):
    """
    Calculate the tripolar pacific index for the Interdecadal Pacific Oscillation (IPO)
    following Henley et al (2015). The IPO is calculated as the average of SST anomalies
    over the central equatorial Pacific (region 2: 10∘ S–10∘ N, 170∘ E–90∘ W) minus the
    average of the SST anomalies in the northwestern (region 1: 25–45∘ N, 140∘ E–145∘ W)
    and southwestern Pacific (region 3: 50–15∘ S, 150∘ E–160∘ W).

    Longitude is assumed to range from 0-360 deg.

    Note typically the IPO index is smoothed in time using a 13-year Chebyshev low-pass
    filter (Henley et al., 2015) or by first applying a 4-year temporal average to the
    sst anomalies (Bilbao at al., 2021).
    """
    region_1 = [140, 215, 25, 45]
    region_2 = [170, 270, -10, 10]
    region_3 = [150, 200, -50, -15]

    ipo = (
        get_lon_lat_average(sst_anom, region_2)
        - (
            get_lon_lat_average(sst_anom, region_1)
            + get_lon_lat_average(sst_anom, region_3)
        )
        / 2
    )
    ipo = ipo.rename({sst_name: "ipo"})
    ipo["ipo"].attrs = dict(long_name="Interdecadal Pacific Oscillation", units="degC")
    return ipo


def calculate_ohc300(temp, depth_dim="depth", temp_name="temp"):
    """
    Calculate the ocean heat content above 300m

    The input DataArray or Dataset is assumed to be in Kelvin

    Parameters
    ----------
    temp : xarray Dataset
        Array of temperature values in Kelvin
    depth_dim : str, optional
        The name of the depth dimension
    temp_name : str, optional
        The name of the temperature variable in temp
    """
    rho0 = 1035.000  # [kg/m^3]
    Cp0 = 3989.245  # [J/kg/K]

    ocean_mask = temp.isel({depth_dim: 0}, drop=True).notnull()
    temp300 = temp.where(temp[depth_dim] <= 300, drop=True).fillna(0)

    # Cast depth coord as float32 to avoid promotion to float64
    temp300 = temp300.assign_coords({depth_dim: temp300[depth_dim].astype(np.float32)})

    ohc300 = rho0 * Cp0 * temp300.integrate(depth_dim)
    ohc300 = ohc300.where(ocean_mask).rename({temp_name: "ohc300"})
    ohc300["ohc300"].attrs = dict(
        long_name="Ocean heat content above 300m", units="J/m^2"
    )
    return ohc300


def ensemble_mean(ds, ensemble_dim="member"):
    """Return the ensemble mean of the input array

    Parameters
    ----------
    ds : xarray Dataset
        Array to take the ensemble mean of
    ensemble_dim : str, optional
        The name of the ensemble dimension
    """
    return ds.mean(ensemble_dim)


def add_CAFE_grid_info(ds):
    """
    Add CAFE grid info to a CAFE dataset that doesn't already have it

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to add grid info to
    """
    atmos_file = PROJECT_DIR / "data/raw/gridinfo/CAFE_atmos_grid.nc"
    ocean_file = PROJECT_DIR / "data/raw/gridinfo/CAFE_ocean_grid.nc"
    atmos_grid = xr.open_dataset(atmos_file)
    ocean_grid = xr.open_dataset(ocean_file)

    atmos = ["area", "zsurf"]  # "latb", "lonb"
    ocean_t = ["area_t", "geolat_t", "geolon_t"]
    ocean_u = ["area_u", "geolat_c", "geolon_c"]

    if ("lat" in ds.dims) | ("lon" in ds.dims):
        ds = ds.assign_coords(atmos_grid[atmos].coords)

    if ("xt_ocean" in ds.dims) | ("yt_ocean" in ds.dims):
        # if "st_ocean" in ds.dims:
        #     ocean_t += ["st_edges_ocean"]
        # if "sw_ocean" in ds.dims:
        #     ocean_t += ["sw_edges_ocean"]
        ds = ds.assign_coords(ocean_grid[ocean_t].coords)

    if ("xu_ocean" in ds.dims) | ("yu_ocean" in ds.dims):
        # if "st_ocean" in ds.dims:
        #     ocean_t += ["st_edges_ocean"]
        # if "sw_ocean" in ds.dims:
        #     ocean_t += ["sw_edges_ocean"]
        ds = ds.assign_coords(ocean_grid[ocean_u].coords)

    return ds


def normalise_by_days_in_month(ds):
    """
    Normalise input array by the number of days in each month

    Parameters
    ----------
    ds : xarray Dataset
        The array to normalise
    """
    # Cast days as float32 to avoid promotion to float64
    return ds / ds["time"].dt.days_in_month.astype(np.float32)


def convert_time_to_lead(
    ds, time_dim="time", time_freq=None, init_dim="init", lead_dim="lead"
):
    """
    Return provided array with time dimension converted to lead time dimension
    and time added as additional coordinate

    Parameters
    ----------
    ds : xarray Dataset
        A dataset with a time dimension
    time_dim : str, optional
        The name of the time dimension
    time_freq : str, optional
        The frequency of the time dimension. If not provided, will try to use
        xr.infer_freq to determine the frequency. This is only used to add a
        freq attr to the lead time coordinate
    init_dim : str, optional
        The name of the initial date dimension in the output
    lead_dim : str, optional
        The name of the lead time dimension in the output
    """
    init_date = ds[time_dim][0].item()
    if time_freq is None:
        time_freq = xr.infer_freq(ds[time_dim])
    lead_time = range(len(ds[time_dim]))
    time_coord = (
        ds[time_dim]
        .rename({time_dim: lead_dim})
        .assign_coords({lead_dim: lead_time})
        .expand_dims({init_dim: [init_date]})
    ).compute()
    dataset = ds.rename({time_dim: lead_dim}).assign_coords(
        {lead_dim: lead_time, init_dim: [init_date]}
    )
    dataset = dataset.assign_coords({time_dim: time_coord})
    dataset[lead_dim].attrs["units"] = time_freq
    return dataset


def truncate_latitudes(ds, dp=10, lat_dim="lat"):
    """
    Return provided array with latitudes truncated to specified dp.

    This is necessary due to precision differences from running forecasts on
    different systems

    Parameters
    ----------
    ds : xarray Dataset
        A dataset with a latitude dimension
    dp : int, optional
        The number of decimal places to truncate at
    lat_dim : str, optional
        The name of the latitude dimension
    """
    for dim in ds.dims:
        if "lat" in dim:
            ds = ds.assign_coords({dim: ds[dim].round(decimals=dp)})
    return ds


def rechunk(ds, chunks):
    """
    Rechunk a dataset

    Parameters
    ----------
    ds : xarray Dataset
        A dataset to be rechunked
    chunks : dict
        Dictionary of {dim: chunksize}
    """
    return ds.chunk(chunks)


def add_attrs(ds, attrs, variable=None):
    """
    Add attributes to a dataset

    Parameters
    ----------
    ds : xarray Dataset
        The data to add attributes to
    attrs : dict
        The attributes to add
    variable : str, optional
        The name of the variable or coordinate to add the attributes to.
        If None, the attributes will be added as global attributes
    """

    if variable is None:
        ds.attrs = attrs
    else:
        ds[variable].attrs = attrs
    return ds


def rename(ds, names):
    """
    Rename all variables etc that have an entry in names

    Parameters
    ----------
    ds : xarray Dataset
        A dataset to be renamed
    names : dict
        Dictionary of {old_name: new_name}
    """
    for k, v in names.items():
        if k in ds:
            ds = ds.rename({k: v})
    return ds


def convert(ds, conversion):
    """
    Convert variables in a dataset according to provided dictionary

    Parameters
    ----------
    ds : xarray Dataset
        A dataset to be converted
    conversion : dict
        Dictionary of {variable: oper} where oper is a dictionary
        specifying the operation and the value. Current possible
        operations are 'multiply_by' and 'add'.
    """
    ds_c = ds.copy()
    for v in conversion.keys():
        if v in ds_c:
            for op, val in conversion[v].items():
                if op == "multiply_by":
                    ds_c[v] *= float(val)
                    if "units" in ds_c[v].attrs:
                        ds_c[v].attrs["units"] = f"{val} * {ds_c[v].attrs['units']}"
                if op == "add":
                    ds_c[v] += float(val)
                    if "units" in ds_c[v].attrs:
                        ds_c[v].attrs["units"] = f"{ds_c[v].attrs['units']} + {val}"
    return ds_c


def mask_period(ds, period):
    """
    Mask times outside of a specified period

    Parameters
    ----------
    ds : xarray Dataset
        The data to mask
    period : iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the period to retain
    """
    # Ensure time is computed
    ds = ds.assign_coords({"time": ds["time"].compute()})

    calendar = ds.time.values.flat[0].calendar
    period = xr.cftime_range(
        period[0],
        period[-1],
        periods=2,
        freq=None,
        calendar=calendar,
    )

    if ("init" in ds.dims) & ("lead" in ds.dims):
        mask = (ds.time >= period[0]) & (ds.time <= period[1])
        return ds.where(mask, drop=True)
    elif "time" in ds.dims:
        return ds.sel(time=slice(period[0], period[1]))
    else:
        raise ValueError("I don't know how to mask the time period for this data")


def anomalise(ds, clim_period):
    """
    Returns the anomalies of ds relative to its climatology over clim_period

    Parameters
    ----------
    ds : xarray Dataset
        The data to anomalise
    clim_period : iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the climatological period
    """
    ds_period = mask_period(ds, clim_period)

    if "time" in ds.dims:
        groupby_dim = "time"
    elif "init" in ds.dims:
        groupby_dim = "init"
    else:
        raise ValueError("I don't know how to compute the anomalies for this data")

    if "member" in ds.dims:
        mean_dim = [groupby_dim, "member"]
    else:
        mean_dim = groupby_dim

    clim = ds_period.groupby(f"{groupby_dim}.month").mean(mean_dim)
    return (ds.groupby(f"{groupby_dim}.month") - clim).drop("month")


def interpolate_to_grid_from_file(ds, file, add_area=True, ignore_degenerate=True):
    import xesmf

    """
    Interpolate to a grid read from a file using xesmf
    file path should be relative to the project directory

    Note, xESMF puts zeros where there is no data to interpolate. Here we
    add an offset to ensure no zeros, mask zeros, and then remove offset
    This hack will potentially do funny things for interpolation methods 
    more complicated than bilinear.
    See https://github.com/JiaweiZhuang/xESMF/issues/15
    
    Parameters
    ----------
    ds : xarray Dataset
        The data to interpolate
    file : str
        Path to a file with the grid to interpolate to
    add_area : bool, optional
        If True (default) add a coordinate for the cell areas
    ignore_degenerate : bool, optional
        If True ESMF will ignore degenerate cells when carrying out
        the interpolation
    """
    file = PROJECT_DIR / file
    ds_out = xr.open_dataset(file)

    C = 1
    ds_rg = ds.copy() + C
    regridder = xesmf.Regridder(
        ds_rg,
        ds_out,
        "bilinear",
        ignore_degenerate=ignore_degenerate,
    )
    ds_rg = regridder(ds_rg, keep_attrs=True)
    ds_rg = ds_rg.where(ds_rg != 0.0) - C

    # Add back in attributes:
    for v in ds_rg.data_vars:
        ds_rg[v].attrs = ds[v].attrs

    if add_area:
        if "area" in ds_out:
            area = ds_out["area"]
        else:
            area = gridarea_cdo(ds_out)
        return ds_rg.assign_coords({"area": area})
    else:
        return ds_rg


def force_to_Julian_calendar(ds, time_dim="time"):
    """
    Hard force calendar of time dimension to Julian

    Parameters
    ----------
    ds : xarray Dataset
        A dataset with a time dimension
    time_dim : str
        The name of the time dimension
    """
    return ds.assign_coords(
        {
            time_dim: xr.cftime_range(
                start=ds.time[0].item().strftime(),
                end=ds.time[-1].item().strftime(),
                freq=xr.infer_freq(ds.time),
                calendar="julian",
            )
        }
    )


def round_to_start_of_month(ds, dim):
    """
    Return provided array with specified time dimension rounded to the start of
    the month

    Parameters
    ----------
    ds : xarray Dataset
        The dataset with a dimension(s) to round
    dim : str
        The name of the dimensions to round
    """
    from xarray.coding.cftime_offsets import MonthBegin

    if isinstance(dim, str):
        dim = [dim]
    for d in dim:
        ds = ds.copy().assign_coords({d: ds[d].compute().dt.floor("D") - MonthBegin()})
    return ds


def coarsen(ds, window_size, start_points=None, dim="time"):
    """
    Coarsen data, applying 'max' to all relevant coords and optionally starting
    at a particular time point in the array

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to coarsen
    start_points : list
        Value(s) of coordinate `dim` to start the coarsening from. If these fall
        outside the range of the coordinate, coarsening starts at the beginning
        of the array
    dim : str, optional
        The name of the dimension to coarsen along
    """
    if start_points is None:
        start_points = [None]

    aux_coords = [c for c in ds.coords if dim in ds[c].dims]
    dss = []
    for start_point in start_points:
        dss.append(
            ds.sel({dim: slice(start_point, None)})
            .coarsen(
                {dim: window_size},
                boundary="trim",
                coord_func={d: "max" for d in aux_coords},
            )
            .mean()
        )
    return xr.concat(dss, dim=dim).sortby(dim)


def rolling_mean(ds, window_size, start_points=None, dim="time"):
    """
    Apply a rolling mean to the data, applying 'max' to all relevant coords and optionally starting
    at a particular time point in the array

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to apply the rolling mean to
    start_points : str or list of str
        Value(s) of coordinate `dim` to start the coarsening from. If these fall
        outside the range of the coordinate, coarsening starts at the beginning
        of the array
    dim : str, optional
        The name of the dimension to coarsen along
    """
    if start_points is None:
        start_points = [None]

    dss = []
    for start_point in start_points:
        rolling_mean = (
            ds.sel({dim: slice(start_point, None)})
            .rolling(
                {dim: window_size},
                min_periods=window_size,
                center=False,
            )
            .mean()
        )

        dss.append(test_rolling.dropna(dim=dim, how="all"))
    return xr.concat(dss, dim=dim).sortby(dim)


def gridarea_cdo(ds):
    """
    Returns the area weights computed using cdo's gridarea function
    Note, this function writes ds to disk, so strip back ds to only what is needed

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to passed to cdo gridarea
    """
    import uuid
    from cdo import Cdo

    infile = uuid.uuid4().hex
    outfile = uuid.uuid4().hex
    ds.to_netcdf(f"./{infile}.nc")
    Cdo().gridarea(input=f"./{infile}.nc", output=f"./{outfile}.nc")
    weights = xr.open_dataset(f"./{outfile}.nc").load()
    os.remove(f"./{infile}.nc")
    os.remove(f"./{outfile}.nc")
    return weights["cell_area"]


def max_chunk_size_MB(ds):
    """
    Get the max chunk size in a dataset
    """

    def size_of_chunk(chunks, itemsize):
        """
        Returns size of chunk in MB given dictionary of chunk sizes
        """
        N = 1
        for value in chunks:
            if not isinstance(value, int):
                value = max(value)
            N = N * value
        return itemsize * N / 1024**2

    chunks = []
    for var in ds.data_vars:
        da = ds[var]
        chunk = da.chunks
        itemsize = da.data.itemsize
        if chunk is None:
            # numpy array
            chunks.append((da.data.size * itemsize) / 1024**2)
        else:
            chunks.append(size_of_chunk(chunk, itemsize))
    return max(chunks)
