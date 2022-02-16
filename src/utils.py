import xarray as xr

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]


def calculate_ohc300(temp, depth_dim="depth"):
    """
    Calculate the ocean heat content above 300m

    The input DataArray or Dataset is assumed to have the variable
    name "temp" and is assumed to be in Kelvin
    """
    rho0 = 1035.000  # [kg/m^3]
    Cp0 = 3989.245  # [J/kg/K]

    ocean_mask = temp.isel({depth_dim: 0}, drop=True).notnull()
    temp300 = temp.where(temp[depth_dim] <= 300, drop=True).fillna(0)
    ohc300 = rho0 * Cp0 * temp300.integrate(depth_dim)
    return ohc300.where(ocean_mask).rename({"temp": "ohc300"})


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


def interpolate_to_grid_from_file(ds, file, add_area=True):
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
    ds = ds.copy() + C
    regridder = xesmf.Regridder(ds, ds_out, "bilinear")
    ds = regridder(ds)
    ds = ds.where(ds != 0.0) - C
    if add_area:
        area = gridarea_cdo(ds_out)
        return ds.assign_coords({"area": area})
    else:
        return ds


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
            .coarsen({dim: 12}, boundary="trim", coord_func={d: "max" for d in aux_coords})
            .mean()
        )
    return xr.concat(dss, dim=dim).sortby(dim)


def gridarea_cdo(ds):
    """
    Returns the area weights computed using cdo's gridarea function
    Note, this function writes ds to disk, so strip back ds to only what is needed
    """
    import os
    from cdo import Cdo

    ds.to_netcdf("./in.nc")
    Cdo().gridarea(input="./in.nc", output="./out.nc")
    weights = xr.open_dataset("./out.nc").load()
    os.remove("./in.nc")
    os.remove("./out.nc")
    return weights["cell_area"]


def estimate_cell_areas(ds, lon_dim="lon", lat_dim="lat"):
    """
    Calculate the area of each grid cell.
    Stolen/adapted from: https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
    """

    from numpy import deg2rad, cos, tan, arctan

    def _earth_radius(lat):
        """Calculate radius of Earth assuming oblate spheroid defined by WGS84"""

        # define oblate spheroid from WGS84
        a = 6378137
        b = 6356752.3142
        e2 = 1 - (b ** 2 / a ** 2)

        # convert from geodecic to geocentric
        # see equation 3-110 in WGS84
        lat = deg2rad(lat)
        lat_gc = arctan((1 - e2) * tan(lat))

        # radius equation
        # see equation 3-107 in WGS84
        return (a * (1 - e2) ** 0.5) / (1 - (e2 * cos(lat_gc) ** 2)) ** 0.5

    R = _earth_radius(ds[lat_dim])

    dlat = deg2rad(ds[lat_dim].diff(lat_dim))
    dlon = deg2rad(ds[lon_dim].diff(lon_dim))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ds[lat_dim]))

    return (dy * dx).broadcast_like(ds[[lon_dim, lat_dim]]).fillna(0)