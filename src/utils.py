import xarray as xr

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def calculate_ohc300(temp, depth_dim="depth", var_name="temp"):
    """
    Calculate the ocean heat content above 300m

    The input DataArray or Dataset is assumed to be in Kelvin
    """
    rho0 = 1035.000  # [kg/m^3]
    Cp0 = 3989.245  # [J/kg/K]

    ocean_mask = temp.isel({depth_dim: 0}, drop=True).notnull()
    temp300 = temp.where(temp[depth_dim] <= 300, drop=True).fillna(0)
    ohc300 = rho0 * Cp0 * temp300.integrate(depth_dim)
    ohc300 = ohc300.where(ocean_mask).rename({var_name: "ohc300"})
    ohc300["ohc300"].attrs = dict(
        long_name="Ocean heat content above 300m", units="J/m^2"
    )
    return ohc300


def add_CAFE_grid_info(ds):
    atmos_file = PROJECT_DIR / "data/raw/gridinfo/CAFE_atmos_grid.nc"
    ocean_file = PROJECT_DIR / "data/raw/gridinfo/CAFE_ocean_grid.nc"
    atmos_grid = xr.open_dataset(atmos_file)
    ocean_grid = xr.open_dataset(ocean_file)

    atmos = ["area", "latb", "lonb", "zsurf"]
    ocean_t = ["area_t", "geolat_t", "geolon_t"]
    ocean_u = ["area_u", "geolat_c", "geolon_c"]

    if ("lat" in ds.dims) | ("lon" in ds.dims):
        ds = ds.assign_coords(atmos_grid.coords)

    if ("xt_ocean" in ds.dims) | ("yt_ocean" in ds.dims):
        if "st_ocean" in ds.dims:
            ocean_t += ["st_edges_ocean"]
        if "sw_ocean" in ds.dims:
            ocean_t += ["sw_edges_ocean"]
        ds = ds.assign_coords(ocean_grid[ocean_t].coords)

    if ("xu_ocean" in ds.dims) | ("yu_ocean" in ds.dims):
        if "st_ocean" in ds.dims:
            ocean_t += ["st_edges_ocean"]
        if "sw_ocean" in ds.dims:
            ocean_t += ["sw_edges_ocean"]
        ds = ds.assign_coords(ocean_grid[ocean_u].coords)

    return ds


def normalise_by_days_in_month(ds):
    """Normalise input array by the number of days in each month"""
    return ds / ds["time"].dt.days_in_month


def convert_time_to_lead(ds, time_dim="time", init_dim="init", lead_dim="lead"):
    """Return provided array with time dimension converted to lead time dimension
    and time added as additional coordinate
    """
    init_date = ds[time_dim].time[0].item()
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
    return dataset


def truncate_latitudes(ds, dp=10):
    """Return provided array with latitudes truncated to specified dp.

    This is necessary due to precision differences from running forecasts on
    different systems
    """
    for dim in ds.dims:
        if "lat" in dim:
            ds = ds.assign_coords({dim: ds[dim].round(decimals=dp)})
    return ds


def rechunk(ds, chunks):
    """Rechunk dataset"""
    return ds.chunk(chunks)


def rename(ds, names):
    """
    Rename all variables etc that have an entry in names
    """
    for k, v in names.items():
        if k in ds:
            ds = ds.rename({k: v})
    return ds


def convert(ds, conversion):
    """
    Convert variables in a dataset according to provided dictionary
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


def anomalise(ds, clim_period):
    """
    Returns the anomalies of ds relative to its climatology over clim_period

    Parameters
    ----------
    ds : xarray Dataset
        The data to anomalise
    clim_period: iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the climatological period
    """
    # Ensure time is computed
    ds = ds.assign_coords({"time": ds["time"].compute()})
    
    calendar = ds.time.values.flat[0].calendar
    clim_period = xr.cftime_range(
        clim_period[0],
        clim_period[-1],
        periods=2,
        freq=None,
        calendar=calendar,
    )
    if ("init" in ds.dims) & ("lead" in ds.dims):
        if "member" in ds.dims:
            mean_dims = ["init", "member"]
        else:
            mean_dims = "init"
        mask = (ds.time >= clim_period[0]) & (ds.time <= clim_period[1])
        clim = ds.where(mask).groupby("init.month").mean(mean_dims)
        return ds.groupby("init.month") - clim
    elif "time" in ds.dims:
        clim = (
            ds.sel(time=slice(clim_period[0], clim_period[1]))
            .groupby("time.month")
            .mean("time")
        )
        return ds.groupby("time.month") - clim
    else:
        raise ValueError("I don't know how to compute the anomalies for this data")


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
    """
    file = PROJECT_DIR / file
    ds_out = xr.open_dataset(file)

    C = 1
    ds_rg = ds.copy() + C
    regridder = xesmf.Regridder(
        ds_rg, ds_out, "bilinear", ignore_degenerate=ignore_degenerate
    )
    ds_rg = regridder(ds_rg)
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


def force_to_Julian_calendar(ds):
    """Force calendar of time dimension to Julian"""
    return ds.assign_coords(
        {
            "time": xr.cftime_range(
                start=ds.time[0].item().strftime(),
                end=ds.time[-1].item().strftime(),
                freq=xr.infer_freq(ds.time),
                calendar="julian",
            )
        }
    )


def round_to_start_of_month(ds, dim):
    """Return provided array with specified time dimension rounded to the start of
    the month
    """
    from xarray.coding.cftime_offsets import MonthBegin

    if isinstance(dim, str):
        dim = [dim]
    for d in dim:
        ds = ds.copy().assign_coords({d: ds[d].compute().dt.floor("D") - MonthBegin()})
    return ds


def coarsen_monthly_to_annual(ds, start_points=None, dim="time"):
    """Coarsen monthly data to annual, applying 'max' to all relevant coords and
    optionally starting at a particular time point in the array
    """
    if start_points is None:
        start_points = [None]

    if isinstance(start_points, str):
        start_points = [start_points]

    aux_coords = [c for c in ds.coords if dim in ds[c].dims]
    dss = []
    for start_point in start_points:
        dss.append(
            ds.sel({dim: slice(start_point, None)})
            .coarsen(
                {dim: 12}, boundary="trim", coord_func={d: "max" for d in aux_coords}
            )
            .mean()
        )
    return xr.concat(dss, dim=dim).sortby(dim)


def gridarea_cdo(ds):
    """
    Returns the area weights computed using cdo's gridarea function
    Note, this function writes ds to disk, so strip back ds to only what is needed
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
