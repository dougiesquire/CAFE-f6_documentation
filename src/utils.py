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


def average_over_lon_lat_box(ds, box, lon_dim="lon", lat_dim="lat"):
    """
    Return the average over a region specified by a range of longitudes and latitudes.

    Longitude is assumed to range from 0-360 deg.

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

    def _get_lat_lon_region(ds, box, lat_dim, lon_dim):
        # Allow for regions that cross 360 deg
        if box[0] > box[1]:
            lon_region = (ds[lon_dim] >= box[0]) | (ds[lon_dim] <= box[1])
        else:
            lon_region = (ds[lon_dim] >= box[0]) & (ds[lon_dim] <= box[1])
        lat_region = (ds[lat_dim] >= box[2]) & (ds[lat_dim] <= box[3])
        region = lon_region & lat_region
        return ds.where(region)

    # Force longitudues to range from 0-360
    ds = ds.assign_coords({lon_dim: (ds[lon_dim] + 360) % 360})

    if (lat_dim in ds.dims) and (lon_dim in ds.dims):
        average_dims = [lat_dim, lon_dim]
        # Allow for regions that cross 360 deg
        if box[0] > box[1]:
            lon_logic_func = np.logical_or
        else:
            lon_logic_func = np.logical_and

        lon_inds = np.where(
            lon_logic_func(ds[lon_dim].values >= box[0], ds[lon_dim].values <= box[1])
        )[0]
        lat_inds = np.where(
            np.logical_and(ds[lat_dim].values >= box[2], ds[lat_dim].values <= box[3])
        )[0]
        region = ds.isel({lon_dim: lon_inds, lat_dim: lat_inds})
    else:
        if (lat_dim in ds.dims) and (lon_dim not in ds.dims):
            average_dims = set([lat_dim, *ds[lon_dim].dims])
        elif (lat_dim not in ds.dims) and (lon_dim in ds.dims):
            average_dims = set([*ds[lat_dim].dims, lon_dim])
        else:
            average_dims = set([*ds[lat_dim].dims, *ds[lon_dim].dims])
        region = _get_lat_lon_region(ds, box, lat_dim, lon_dim)
    return region.weighted(ds["area"].fillna(0)).mean(dim=average_dims)


def calculate_nino34(sst_anom, sst_name="sst"):
    """
    Calculate the NINO3.4 index. The NINO3.4 index is calculated as the spatial average
    of SST anomalies over the tropical Pacific region (5∘S–5∘N and 170–120∘ W).

    Parameters
    ----------
    sst_anom : xarray Dataset
        Array of sst anomalies
    sst_name : str, optional
        The name of the sst variable in sst_anom
    """

    box = [190.0, 240.0, -5.0, 5.0]
    nino34 = average_over_lon_lat_box(sst_anom, box)
    nino34 = nino34.rename({sst_name: "nino34"})
    nino34["nino34"].attrs = dict(long_name="ENSO Nino 3.4 Index", units="degC")
    return nino34


def calculate_dmi(sst_anom, sst_name="sst"):
    """
    Calculate the Dipole Mode Index (DMI) for the Indian Ocean Dipole. The DMI is
    calculated as the difference between the spatial averages of SST anomalies over
    two regions of the tropical Indian Ocean: (10°S-10°N and 50°E-70°E) and
    (10°S-0°S and 90°E-110°E).

    Parameters
    ----------
    sst_anom : xarray Dataset
        Array of sst anomalies
    sst_name : str, optional
        The name of the sst variable in sst_anom
    """
    region_W = [50.0, 70.0, -10.0, 10.0]
    region_E = [90.0, 110.0, -10.0, 0.0]

    dmi = average_over_lon_lat_box(sst_anom, region_W) - average_over_lon_lat_box(
        sst_anom, region_E
    )
    dmi = dmi.rename({sst_name: "dmi"})
    dmi["dmi"].attrs = dict(long_name="IOD Dipole Mode Index", units="degC")
    return dmi


def calculate_sam(
    slp, clim_period, groupby_dim="time", slp_name="slp", lon_dim="lon", lat_dim="lat"
):
    """
    Calculate the Southern Annular Mode index from monthly data as defined by Gong, D.
    and Wang, S., 1999. The SAM index is defined as the difference between the normalized
    monthly zonal mean sea level pressure at 40∘S and 65∘S.

    Parameters
    ----------
    slp : xarray Dataset
        Array of sea level pressures
    clim_period : iterable
        Size 2 iterable containing strings indicating the start and end dates of the
        climatological period used to normalise the SAM index
    groupby_dim : str
        The dimension to compute the normalisation over
    slp_name : str, optional
        The name of the slp variable in the input slp Dataset
    lon_dim : str, optional
        The name of the longitude dimension
    lat_dim : str, optional
        The name of the latitude dimension
    """

    def _normalise_sam(group, clim_group):
        """Return the anomalies normalize by their standard deviation"""
        this_month = group[groupby_dim].dt.month.values[0]
        grouped_months, _ = zip(*list(clim_group))
        clim_group_this_month = list(clim_group)[grouped_months.index(this_month)][1]
        average_dim = [groupby_dim, "member"] if "member" in group.dims else groupby_dim
        return (
            group - clim_group_this_month.mean(average_dim)
        ) / clim_group_this_month.std(average_dim)

    slp_40 = slp.interp({lat_dim: -40}).mean(lon_dim)
    slp_65 = slp.interp({lat_dim: -65}).mean(lon_dim)

    slp_40_clim_period = keep_period(slp_40, clim_period)
    slp_65_clim_period = keep_period(slp_65, clim_period)

    slp_40_group = slp_40.groupby(groupby_dim + ".month")
    slp_40_clim_period_group = slp_40_clim_period.groupby(groupby_dim + ".month")
    slp_65_group = slp_65.groupby(groupby_dim + ".month")
    slp_65_clim_period_group = slp_65_clim_period.groupby(groupby_dim + ".month")

    norm_40 = slp_40_group.map(_normalise_sam, clim_group=slp_40_clim_period_group)
    norm_65 = slp_65_group.map(_normalise_sam, clim_group=slp_65_clim_period_group)

    sam = norm_40 - norm_65
    sam = sam.rename({slp_name: "sam"})
    sam["sam"].attrs = dict(long_name="Southern Annular Mode index", units="-")
    return sam


def calculate_nao(
    slp, clim_period, groupby_dim="time", slp_name="slp", lon_dim="lon", lat_dim="lat"
):
    """
    Calculate the Northern Atlantic Oscillation index from monthly data as defined by
    Jianping, L. & Wang, J. X. L. (2003). The NAO index is defined as the difference
    between the normalized monthly mean sea level pressure at 35∘N and 65∘N, averaged
    over the zonal band spanning 80◦W–30◦E

    Parameters
    ----------
    slp : xarray Dataset
        Array of sea level pressures
    clim_period : iterable
        Size 2 iterable containing strings indicating the start and end dates of the
        climatological period used to normalise the NAO index
    groupby_dim : str
        The dimension to compute the normalisation over
    slp_name : str, optional
        The name of the slp variable in the input slp Dataset
    lon_dim : str, optional
        The name of the longitude dimension
    lat_dim : str, optional
        The name of the latitude dimension
    """

    def _get_lon_band(ds, lon_range, lon_dim):
        """Return the average over a longitudinal band"""
        if lon_range[0] > lon_range[1]:
            logic_func = np.logical_or
        else:
            logic_func = np.logical_and
        lon_inds = np.where(
            logic_func(
                ds[lon_dim].values >= lon_range[0], ds[lon_dim].values <= lon_range[1]
            )
        )[0]
        return ds.isel({lon_dim: lon_inds}).mean(lon_dim)

    def _normalise_nao(group, clim_group):
        """Return the anomalies normalize by their standard deviation"""
        this_month = group[groupby_dim].dt.month.values[0]
        grouped_months, _ = zip(*list(clim_group))
        clim_group_this_month = list(clim_group)[grouped_months.index(this_month)][1]
        average_dim = [groupby_dim, "member"] if "member" in group.dims else groupby_dim
        return (
            group - clim_group_this_month.mean(average_dim)
        ) / clim_group_this_month.std(average_dim)

    # Force longitudues to range from 0-360
    slp = slp.assign_coords({lon_dim: (slp[lon_dim] + 360) % 360})

    lon_band = [280, 30]
    slp_35 = _get_lon_band(slp.interp({lat_dim: 35}), lon_band, lon_dim)
    slp_65 = _get_lon_band(slp.interp({lat_dim: 65}), lon_band, lon_dim)

    slp_35_clim_period = keep_period(slp_35, clim_period)
    slp_65_clim_period = keep_period(slp_65, clim_period)

    slp_35_group = slp_35.groupby(groupby_dim + ".month")
    slp_35_clim_period_group = slp_35_clim_period.groupby(groupby_dim + ".month")
    slp_65_group = slp_65.groupby(groupby_dim + ".month")
    slp_65_clim_period_group = slp_65_clim_period.groupby(groupby_dim + ".month")

    norm_35 = slp_35_group.map(_normalise_nao, clim_group=slp_35_clim_period_group)
    norm_65 = slp_65_group.map(_normalise_nao, clim_group=slp_65_clim_period_group)

    nao = norm_35 - norm_65
    nao = nao.rename({slp_name: "nao"})
    nao["nao"].attrs = dict(long_name="Northern Atlantic Oscillation index", units="-")
    return nao


def calculate_amv(sst_anom, sst_name="sst"):
    """
    Calculate the Atlantic Multi-decadal Variability (AMV)--also known as the Atlantic
    Multi-decadal Oscillation (AMO)--according to Trenberth and Shea (2006). The AMV
    is calculated as the spatial average of SST anomalies over the North Atlantic
    (Equator–60∘ N and 80–0∘ W) minus the spatial average of SST anomalies averaged from
    60∘ S to 60∘ N.

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

    north_atlantic_box = [280.0, 360.0, 0.0, 60.0]
    global_box = [0.0, 360.0, -60.0, 60.0]

    amv = average_over_lon_lat_box(
        sst_anom, north_atlantic_box
    ) - average_over_lon_lat_box(sst_anom, global_box)
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

    Note typically the IPO index is smoothed in time using a 13-year Chebyshev low-pass
    filter (Henley et al., 2015) or by first applying a 4-year temporal average to the
    sst anomalies (Bilbao at al., 2021).
    """
    region_1 = [140.0, 215.0, 25.0, 45.0]
    region_2 = [170.0, 270.0, -10.0, 10.0]
    region_3 = [150.0, 200.0, -50.0, -15.0]

    ipo = (
        average_over_lon_lat_box(sst_anom, region_2)
        - (
            average_over_lon_lat_box(sst_anom, region_1)
            + average_over_lon_lat_box(sst_anom, region_3)
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

    # Cast as float64 since big numbers
    ohc300 = rho0 * Cp0 * temp300.integrate(depth_dim).astype(np.float64)
    ohc300 = ohc300.where(ocean_mask).rename({temp_name: "ohc300"})
    ohc300["ohc300"].attrs = dict(
        long_name="Ocean heat content above 300m", units="J/m^2"
    )
    return ohc300


def calculate_wind_speed(u_v, u_name, v_name, lon_dim="lon", lat_dim="lat"):
    """
    Calculate the wind speed

    Parameters
    ----------
    u_v : xarray Dataset
        Dataset containing the longitudinal and latitudinal components of the wind
    u_name : str
        The name of the u-velocity variable in u
    v_name : str
        The name of the v-velocity variable in v
    lon_dim : str, optional
        The name of the longitude dimension for u and v
    lat_dim : str, optional
        The name of the latitude dimension for u and v
    """
    V = np.sqrt(u_v[u_name] ** 2 + u_v[v_name] ** 2).to_dataset(name="V_tot")
    V["V_tot"].attrs = dict(
        long_name="Total wind speed", units=f"{u_v[u_name].attrs['units']}"
    )
    return V

def calculate_tmean_from_tmin_tmax(ds, tmin_name="tmin", tmax_name="tmax", tmean_name="tmean"):
    """
    Estimate tmean as the average of tmin and tmax

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing tmin and tmax variables
    tmin_name : str
        The name of the tmin variable
    tmax_name : str
        The name of the tmax variable
    tmean_name : str
        The name of the output tmean variable
    """
    tmean = ((ds[tmin_name] + ds[tmax_name]) / 2).to_dataset(name=tmean_name)
    tmean[tmean_name].attrs["long_name"] = "Daily mean air temperature"
    return tmean


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


def drop_Feb_29(ds, time_dim="time"):
    """
    Drop all occurences of Feb 29th
    
    Parameters
    ----------
    ds : xarray Dataset
        A dataset with a time dimension
    time_dim : str, optional
        The name of the time dimension
    """
    keep = np.logical_not((ds[time_dim].dt.month == 2) & (ds[time_dim].dt.day == 29))
    return ds.where(keep, drop=True)


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


def keep_period(ds, period):
    """
    Keep only times outside of a specified period

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


def _get_groupby_and_mean_dims(ds):
    """
    Get the groupby and averaging dimensions for performing operations like
    calculating anomalies and percentile thresholds
    """
    if "time" in ds.dims:
        groupby_dim = "time"
    elif "init" in ds.dims:
        groupby_dim = "init"
    else:
        raise ValueError("I don't know how to compute the percentiles for this data")

    if "member" in ds.dims:
        mean_dim = [groupby_dim, "member"]
    else:
        mean_dim = groupby_dim

    return groupby_dim, mean_dim


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
    ds_period = keep_period(ds, clim_period)

    groupby_dim, mean_dim = _get_groupby_and_mean_dims(ds)

    clim = ds_period.groupby(f"{groupby_dim}.month").mean(mean_dim)
    return (ds.groupby(f"{groupby_dim}.month") - clim).drop("month")


def calculate_percentile_thresholds(ds, percentile, percentile_period):
    """
    Returns the percentile values of ds over a provided period

    Parameters
    ----------
    ds : xarray Dataset
        The data to calculate the percentiles
    percentile : float
        The percentile to calculate
    percentile_period : iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the period over which to calculate the percentile thresholds
    """
    ds_period = keep_period(ds, percentile_period)

    groupby_dim, mean_dim = _get_groupby_and_mean_dims(ds)

    return ds_period.groupby(f"{groupby_dim}.month").quantile(
        q=percentile, dim=mean_dim
    )


def over_percentile_threshold(ds, percentile, percentile_period):
    """
    Find which values in the input array are over a specified percentile
    calculated over a specified period. Returns a boolean array with True
    where values are over the specified percentile and False elsewhere.

    Parameters
    ----------
    ds : xarray Dataset
        The data threshold based in it's percentiles
    percentile : float
        The percentile use to threshold the data
    percentile_period : iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the period over which to calculate the percentile thresholds
    """
    groupby_dim, _ = _get_groupby_and_mean_dims(ds)

    percentile_thresholds = calculate_percentile_thresholds(
        ds, percentile, percentile_period
    )

    return (ds.groupby(f"{groupby_dim}.month") > percentile_thresholds).drop("month")


def under_percentile_threshold(ds, percentile, percentile_period):
    """
    Find which values in the input array are under a specified percentile
    calculated over a specified period. Returns a boolean array with True
    where values are under the specified percentile and False elsewhere.

    Parameters
    ----------
    ds : xarray Dataset
        The data threshold based in it's percentiles
    percentile : float
        The percentile use to threshold the data
    percentile_period : iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the period over which to calculate the percentile thresholds
    """
    groupby_dim, _ = _get_groupby_and_mean_dims(ds)

    percentile_thresholds = calculate_percentile_thresholds(
        ds, percentile, percentile_period
    )

    return (ds.groupby(f"{groupby_dim}.month") < percentile_thresholds).drop("month")


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
                freq=xr.infer_freq(ds.time[:3]),
                calendar="julian",
            )
        }
    )


def round_to_start_of_day(ds, dim):
    """
    Return provided array with specified time dimension rounded to the start of
    the day

    Parameters
    ----------
    ds : xarray Dataset
        The dataset with a dimension(s) to round
    dim : str
        The name of the dimensions to round
    """
    if isinstance(dim, str):
        dim = [dim]
    for d in dim:
        ds = ds.copy().assign_coords({d: ds[d].compute().dt.floor("D")})
    return ds


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
    Apply a rolling mean to the data, applying 'max' to all relevant coords and
    optionally starting at a particular time point in the array

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

        dss.append(rolling_mean.dropna(dim=dim, how="all"))
    result = xr.concat(dss, dim=dim).sortby(dim)

    # For reasons I don't understand, rolling sometimes promotes float32 to float64
    return xr.merge([result[var].astype(ds[var].dtype) for var in ds.data_vars])


def resample(ds, freq, start_points=None, dim="time"):
    """
    Resample data to a different temporal frequency by taking the mean
    over all values at the downsampled frequency and optionally starting
    at a particular time point in the array

    Parameters
    ----------
    ds : xarray Dataset
        The dataset to resample
    freq : str
        Resample frequency expressed using pandas offset alias
    start_points : str or list of str
        Value(s) of coordinate `dim` to start the resampling from. If these fall
        outside the range of the coordinate, resampling starts at the beginning
        of the array
    dim : str, optional
        The name of the time dimension to resample along
    """
    if start_points is None:
        start_points = [None]

    dss = []
    for start_point in start_points:
        dss.append(
            ds.sel({dim: slice(start_point, None)}).resample({dim: freq}).mean(dim)
        )

    return xr.concat(dss, dim=dim).sortby(dim)


def get_region_masks_from_shp(ds, shapefile, header):
    """
    Extract region masks according to a shapefile

    Parameters
    ----------
    ds : xarray Dataset
        The array with the grid to build the masks for
    shapefile : str
        The path to the shapefile to use
    header : str
        Name of the shapefile column to use to name the regions
    """
    import geopandas
    import regionmask

    shapes = geopandas.read_file(shapefile)
    mask = regionmask.mask_3D_geopandas(shapes, ds.lon, ds.lat)
    return mask.assign_coords({"region": shapes[header].to_list()})


def average_over_NRM_super_clusters(ds):
    """
    Average the provided array over the NRM super cluster regions

    Parameters
    ----------
    ds : xarray Dataset
        The array to average over the NRM super cluster regions
    """
    shapefile = PROJECT_DIR / "data/raw/NRM_super_clusters/NRM_super_clusters.shp"
    header = "label"
    masks = get_region_masks_from_shp(ds, shapefile, header)
    masks = xr.concat(
        [masks, masks.sum("region").assign_coords({"region": "Australia"})],
        dim="region",
    )
    return ds.where(masks).weighted(ds["area"]).mean(["lon", "lat"])


def mask_CAFEf6_reduced_dt(ds):
    """
    Mask out the ensemble members of CAFE-f6 that were run with a reduced timestep
    since reducing the timestep was found to produce a different model equilibrium

    Parameters
    ----------
    ds : xarray Dataset
        The CAFE-f6 data to mask
    """
    mask_file = PROJECT_DIR / "data/raw/CAFEf6/CAFE-f6_dt_atmos.nc"
    mask = xr.open_dataset(mask_file)["dt_atmos"] >= 1800
    mask = mask.rename({"init_date": "init", "ensemble": "member"})
    mask = mask.assign_coords({"init": mask.init.dt.floor("D")})
    return ds.where(mask)


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


def add_area_using_cdo_gridarea(ds, lon_dim="lon", lat_dim="lat"):
    """
    Add a area coordinate to the provided dataset containing the cell areas
    estimated by cdo's gridarea function

    Parameters
    ----------
    ds : xarray Dataset
        The data to use to estimate the cell areas
    lon_dim : str, optional
        The name of the longitude dimension on ds
    lat_dim : str, optional
        The name of the latitude dimension on ds
    """
    other_dims = set(ds.dims) - set([lon_dim, lat_dim])
    area = gridarea_cdo(ds.isel({d: 0 for d in other_dims}).load())
    return ds.assign_coords({"area": area})


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
