# Helper functions for opening data in a common format
__all__ = [
    "CAFEf6",
    "CAFEf5",
    "CAFE60v1",
    "CAFEh0",
    "JRA55",
    "HadISST",
]

import glob

import warnings

import xarray as xr

from .utils import (
    force_to_Julian_calendar,
    round_to_start_of_month,
    convert_time_to_lead,
    truncate_latitudes,
)


PATHS = {
    "CAFEf6": "/g/data/xv83/dcfp/CAFE-f6/",
    "CAFEf5": "/g/data/xv83/dcfp/CAFE-f5/",
    "CAFE60v1": "/g/data/xv83/dcfp/CAFE60v1/",
    "CAFEh0": "/g/data/xv83/users/ds0092/data/CAFE/historical/WIP/",
    "JRA55": "/g/data/xv83/reanalyses/JRA55/",
    "HadISST": "/g/data/xv83/reanalyses/HadISST/",
    "EN422": "/g/data/xv83/reanalyses/EN.4.2.2/",
    "CanESM5": "/g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast/",
}

JRA55_VARIABLE_TRANSLATION = {
    "t_ref": "TMP_GDS0_HTGL",
    "precip": "TPRAT_GDS0_SFC",
}

EN422_VARIABLE_TRANSLATION = {
    "temp": "temperature",
    "salt": "salinity",
}

CMIP_VARIABLE_TRANSLATION = {
    "t_ref": "tas",
    "precip": "pr",
    "sst": "tos"
}


def _normalise_variables(ds):
    """Rescale CAFE variables"""

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


def _preprocess_forecast(ds):
    """Preprocessing steps for CAFE forecast data"""
    ds = convert_time_to_lead(ds)
    # Truncate lats so that forecasts run on different systems can be stacked
    if "lat" in ds.dims:
        ds = truncate_latitudes(ds)
    return ds


def _maybe_translate_variables(variables, translation_dict):
    """Returns variable names in EN.4.2.2 data and dictionary for translation"""
    from operator import itemgetter

    translated_variables = []
    translation = {}
    for v in variables:
        try:
            translated_variables.append(translation_dict[v])
            translation[translation_dict[v]] = v
        except KeyError as exception:
            warnings.warn(
                f"No translation exists between {exception} and its name in the target dataset."
            )
            translated_variables.append(v)

    return translated_variables, translation


def CAFEf6(realm, variables, preprocess=None):
    """Open CAFE-f6 forecast data"""

    if isinstance(variables, str):
        variables = [variables]

    files = sorted(
        glob.glob(f"{PATHS['CAFEf6']}/c5-d60-pX-f6-????1101/{realm}.zarr.zip")
    )  # Skip May starts
    ds = xr.open_mfdataset(
        files,
        compat="override",
        preprocess=lambda x: preprocess(_preprocess_forecast(x)),
        engine="zarr",
        coords="minimal",
        parallel=True,
    )[variables]

    ds = round_to_start_of_month(ds, dim=["init", "time"])
    ds = _normalise_variables(ds)
    return ds


def CAFEf5(realm, variables, preprocess=None):
    """Open CAFE-f5 forecast data"""

    if isinstance(variables, str):
        variables = [variables]

    ds = xr.open_dataset(
        f"{PATHS['CAFEf5']}/NOV/{realm}.zarr.zip", engine="zarr", chunks={}
    )[
        variables
    ]  # Skip May starts
    ds = ds.rename({"init_date": "init", "lead_time": "lead"})

    # Append 2020 forecast from CAFE-f6
    ds_2020 = xr.open_dataset(
        f"{PATHS['CAFEf6']}/c5-d60-pX-f6-20201101/{realm}.zarr.zip",
        engine="zarr",
        chunks={},
    )[variables]
    ds_2020 = _preprocess_forecast(ds_2020)

    ds = ds.assign_coords({"time": ds["time"].compute()})  # Required for concat below
    ds = xr.concat([ds, ds_2020.isel(ensemble=range(10))], dim="init")

    if preprocess is not None:
        ds = preprocess(ds)
    ds = round_to_start_of_month(ds, dim=["init", "time"])
    ds = _normalise_variables(ds)
    return ds


def CAFE60v1(realm, variables):
    """Open CAFE60v1 data"""

    if isinstance(variables, str):
        variables = [variables]

    ds = xr.open_dataset(
        f"{PATHS['CAFE60v1']}/{realm}.zarr.zip", engine="zarr", chunks={}
    )[variables]

    ds = truncate_latitudes(ds)
    ds = round_to_start_of_month(ds, dim="time")
    ds = _normalise_variables(ds)
    return ds


def CAFEh0(realm, variables):
    """Open CAFE historical run data"""

    if isinstance(variables, str):
        variables = [variables]

    hist = xr.open_dataset(
        f"{PATHS['CAFEh0']}/c5-d60-pX-hist-19601101/ZARR/{realm}.zarr.zip",
        engine="zarr",
        chunks={},
    )[variables]

    ctrl = xr.open_dataset(
        f"{PATHS['CAFEh0']}/c5-d60-pX-ctrl-19601101/ZARR/{realm}.zarr.zip",
        engine="zarr",
        chunks={},
    )[variables]

    hist = truncate_latitudes(hist)
    ctrl = truncate_latitudes(ctrl)
    hist = round_to_start_of_month(hist, dim="time")
    ctrl = round_to_start_of_month(ctrl, dim="time")
    hist = _normalise_variables(hist)
    ctrl = _normalise_variables(ctrl)

    drift = (
        ctrl.mean("ensemble").groupby("time.month").map(lambda x: x - x.mean(["time"]))
    )
    return hist - drift


def JRA55(realm, variables):
    """Open JRA55 data using CAFE variable naming"""

    if isinstance(variables, str):
        variables = [variables]

    JRA55_variables, rename = _maybe_translate_variables(
        variables, JRA55_VARIABLE_TRANSLATION
    )
    ds = xr.open_dataset(
        f"{PATHS['JRA55']}/{realm}.zarr.zip",
        engine="zarr",
        chunks={},
        use_cftime=True,
    )[JRA55_variables]
    ds = ds.rename({**rename, "initial_time0_hours": "time"})
    ds = force_to_Julian_calendar(ds)
    ds = truncate_latitudes(ds)

    return ds


def HadISST():
    """Open HadISST data"""

    ds = xr.open_dataset(
        f"{PATHS['HadISST']}/ocean_month.zarr",
        engine="zarr",
        chunks={},
        use_cftime=True,
    )[["sst"]]

    ds = ds.where(ds > -1000)
    ds = ds.rename({"longitude": "lon", "latitude": "lat"})

    ds = round_to_start_of_month(ds, dim="time")
    ds = force_to_Julian_calendar(ds)
    return ds


def EN422(variables):
    """Open EN.4.2.2 data"""

    if isinstance(variables, str):
        variables = [variables]

    EN422_variables, rename = _maybe_translate_variables(
        variables, EN422_VARIABLE_TRANSLATION
    )

    ds = xr.open_mfdataset(
        f"{PATHS['EN422']}/*.nc",
        parallel=True,
        use_cftime=True,
    )[EN422_variables]
    ds = ds.rename(rename)
    ds = round_to_start_of_month(ds, dim="time")
    ds = force_to_Julian_calendar(ds)

    return ds


def CanESM5(realm, variables):
    import dask

    @dask.delayed
    def _open_CanESM5_delayed(y, e, v):
        file = f"{PATHS['CanESM5']}/s{y-1}-r{e}i1p2f1/{realm}/{v}/gn/v20190429/{v}_{realm}_CanESM5_dcppA-hindcast_s{y-1}-r{e}i1p2f1_gn_{y}01-{y+9}12.nc"
        ds = xr.open_dataset(file, chunks={})[v]
        return ds

    def _open_CanESM5(y, e, v):
        var_data = _open_CanESM5_delayed(y, e, v).data

        # Tell Dask the delayed function returns an array, and the size and type of that array
        return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

    if isinstance(variables, str):
        variables = [variables]

    CMIP_variables, rename = _maybe_translate_variables(
        variables, CMIP_VARIABLE_TRANSLATION
    )

    years = range(1981, 2018)  # Ocean files end in 2017
    ensembles = range(1, 40 + 1)

    ds = []
    for v in CMIP_variables:
        ds0 = xr.open_dataset(
            f"{PATHS['CanESM5']}/s{years[0]-1}-r{ensembles[0]}i1p2f1/{realm}/{v}/gn/v20190429/{v}_{realm}_CanESM5_dcppA-hindcast_s{years[0]-1}-r{ensembles[0]}i1p2f1_gn_{years[0]}01-{years[0]+9}12.nc",
            chunks={},
        )
        d0 = convert_time_to_lead(ds0)[v]

        delayed = []
        for y in years:
            delayed.append(
                dask.array.stack([_open_CanESM5(y, e, v) for e in ensembles], axis=0)
            )
        delayed = dask.array.stack(delayed, axis=0)

        init = xr.cftime_range(
            str(years[0]), str(years[-1]), freq="YS", calendar="julian"
        )
        member = ensembles
        time = [
            xr.cftime_range(i, periods=120, freq="MS", calendar="julian") for i in init
        ]
        ds.append(
            xr.DataArray(
                delayed,
                dims=["init", "member", *d0.dims],
                coords={
                    "member": ensembles,
                    "init": init,
                    **d0.coords,
                    "time": (["init", "lead"], time),
                },
                attrs=d0.attrs,
            ).to_dataset(name=v)
        )
    ds = xr.merge(ds).rename(rename)
    ds = _normalise_variables(ds)

    return ds.compute()