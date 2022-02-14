# Helper functions for opening data in a common format

import glob

import dask
import xarray as xr

import yaml
from functools import reduce, partial

from . import utils


def _load_config(name):
    """Load a config .yaml file for a specified dataset"""
    with open(name, "r") as reader:
        return yaml.load(reader, Loader=yaml.BaseLoader)


def _maybe_translate_variables(variables, translation_dict):
    """
    Translate variables using provided dictionary where possible
    """
    translated_variables = {}
    for realm, var in variables.items():
        translated_variables[realm] = []
        for v in var:
            try:
                translated_variables[realm].append(translation_dict[v])
            except KeyError as exception:
                translated_variables[realm].append(v)
    return translated_variables


def _maybe_rename(ds, rename):
    """
    Rename all variables etc that have an entry in rename
    """
    for k, v in rename.items():
        if v in ds:
            ds = ds.rename({v: k})
    return ds


def _normalise(ds, norm_dict):
    """
    Rescale variables in a dataset according to provided dictionary
    """
    for v in norm_dict.keys():
        if v in ds:
            ds[v] = float(norm_dict[v]) * ds[v]
    return ds


def _composite_function(function_dict):
    """
    Return a composite function of all functions specified in a processing
        step of a config .yaml
    """

    def composite(*funcs):
        def compose(f, g):
            return lambda x: g(f(x))

        return reduce(compose, funcs, lambda x: x)

    funcs = []
    for fn in function_dict.keys():
        kws = function_dict[fn]
        kws = {} if kws == "" else kws
        funcs.append(partial(getattr(utils, fn), **kws))

    return composite(*funcs)


def open_dataset(config):
    """
    Open a dataset according to specifications in a config file
    """
    cfg = _load_config(config)

    dataset = cfg["name"]

    if "variables" in cfg:
        if isinstance(cfg["variables"], list):
            variables = {None: cfg["variables"]}
        else:
            variables = cfg["variables"]
    else:
        raise ValueError(
            "No variables are listed in the config or were provided to this function"
        )

    if "rename" in cfg:
        variables = _maybe_translate_variables(variables, cfg["rename"])

    if "preprocess" in cfg:
        preprocess = _composite_function(cfg["preprocess"])
    else:
        preprocess = None

    ds = []
    for realm, var in variables.items():
        ds.append(getattr(_open, dataset)(cfg["path"], realm, var, preprocess))
    ds = xr.merge(ds)

    if "rename" in cfg:
        ds = _maybe_rename(ds, cfg["rename"])

    if "normalise" in cfg:
        ds = _normalise(ds, cfg["normalise"])

    if "postprocess" in cfg:
        ds = _composite_function(cfg["postprocess"])(ds)

    return ds


class _open:
    """
    Class containing the dataset-specific code for opening each available dataset
    """

    def JRA55(path, realm, variables, _):
        """Open JRA55 variables from specified realm"""
        return xr.open_dataset(
            f"{path}/{realm}.zarr.zip",
            engine="zarr",
            chunks={},
            use_cftime=True,
        )[variables]

    def HadISST(path, realm, variables, _):
        """Open HadISST variables from specified realm"""
        ds = xr.open_dataset(
            f"{path}/{realm}.zarr",
            engine="zarr",
            chunks={},
            use_cftime=True,
        )[variables]
        return ds.where(ds > -1000)

    def EN422(path, _, variables, __):
        """Open EN.4.2.2 variables"""
        return xr.open_mfdataset(
            f"{path}/*.nc",
            parallel=True,
            use_cftime=True,
        )[variables]

    def CAFEf6(path, realm, variables, preprocess):
        """Open CAFE-f6 variables from specified realm applying preprocess prior to
        concanenating forecasts
        """
        files = sorted(
            glob.glob(f"{path}/c5-d60-pX-f6-????1101/{realm}.zarr.zip")
        )  # Skip May starts

        return xr.open_mfdataset(
            files,
            compat="override",
            preprocess=preprocess,
            engine="zarr",
            coords="minimal",
            parallel=True,
        )[variables]

    def CAFEf5(path, realm, variables, _):
        """Open CAFE-f5 variables from specified realm, including appending first
        10 members of CAFE-f6 for 2020 forecast
        """
        ds = xr.open_dataset(f"{path}/NOV/{realm}.zarr.zip", engine="zarr", chunks={})[
            variables
        ]

        # Append 2020 forecast from CAFE-f6
        cfg_f6 = _load_config("CAFEf6")
        ds_2020 = xr.open_dataset(
            f"{cfg_f6['path']}/c5-d60-pX-f6-20201101/{realm}.zarr.zip",
            engine="zarr",
            chunks={},
        )[variables].isel(ensemble=range(10))
        ds_2020 = utils.convert_time_to_lead(
            ds_2020, init_dim="init_date", lead_dim="lead_time"
        )
        ds_2020 = utils.truncate_latitudes(ds_2020)

        ds = ds.assign_coords(
            {"time": ds["time"].compute()}
        )  # Required for concat below
        return xr.concat([ds, ds_2020], dim="init_date")

    def CAFE60v1(path, realm, variables, _):
        """Open CAFE60v1 variables from specified realm"""
        return xr.open_dataset(f"{path}/{realm}.zarr.zip", engine="zarr", chunks={})[
            variables
        ]

    def CAFE_hist(path, realm, variables, _):
        """Open CAFE historical run variables from specified realm"""
        hist = xr.open_dataset(
            f"{path}/c5-d60-pX-hist-19601101/ZARR/{realm}.zarr.zip",
            engine="zarr",
            chunks={},
        )[variables]

        ctrl = xr.open_dataset(
            f"{path}/c5-d60-pX-ctrl-19601101/ZARR/{realm}.zarr.zip",
            engine="zarr",
            chunks={},
        )[variables]

        hist = utils.truncate_latitudes(hist)
        ctrl = utils.truncate_latitudes(ctrl)

        drift = (
            ctrl.mean("ensemble")
            .groupby("time.month")
            .map(lambda x: x - x.mean(["time"]))
        )
        return hist - drift

    def CanESM5(path, realm, variables, _):
        """Open CanESM5 dcppA-hindcast variables from specified realm"""

        def _CanESM5_file(y, m, v):
            version = "v20190429"
            return f"{path}/s{y-1}-r{m}i1p2f1/{realm}/{v}/gn/{version}/{v}_{realm}_CanESM5_dcppA-hindcast_s{y-1}-r{m}i1p2f1_gn_{y}01-{y+9}12.nc"

        @dask.delayed
        def _open_CanESM5_delayed(y, m, v):
            file = _CanESM5_file(y, m, v)
            ds = xr.open_dataset(file, chunks={})[v]
            return ds

        def _open_CanESM5(y, m, v, d0):
            var_data = _open_CanESM5_delayed(y, m, v).data
            return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

        years = range(1981, 2018)  # CanESM5 ocean files end in 2017
        members = range(1, 40 + 1)

        ds = []
        for v in variables:
            f0 = _CanESM5_file(years[0], members[0], v)
            d0 = utils.convert_time_to_lead(xr.open_dataset(f0, chunks={}))[v]

            delayed = []
            for y in years:
                delayed.append(
                    dask.array.stack(
                        [_open_CanESM5(y, m, v, d0) for m in members], axis=0
                    )
                )
            delayed = dask.array.stack(delayed, axis=0)

            init = xr.cftime_range(
                str(years[0]), str(years[-1]), freq="YS", calendar="julian"
            )
            time = [
                xr.cftime_range(i, periods=120, freq="MS", calendar="julian")
                for i in init
            ]
            ds.append(
                xr.DataArray(
                    delayed,
                    dims=["init", "member", *d0.dims],
                    coords={
                        "member": members,
                        "init": init,
                        **d0.coords,
                        "time": (["init", "lead"], time),
                    },
                    attrs=d0.attrs,
                ).to_dataset(name=v)
            )
        return xr.merge(ds).compute()

    def CanESM5_hist(path, realm, variables, _):
        """Open CanESM5 historical variables from specified realm"""

        @dask.delayed
        def _open_CanESM5_hist_delayed(f, v):
            ds = xr.open_dataset(f, chunks={})[v]
            return ds

        def _open_CanESM5_hist(f, v):
            var_data = _open_CanESM5_hist_delayed(f, v).data
            return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

        ds = []
        members = range(1, 40 + 1)
        for v in variables:
            files = sorted(
                glob.glob(
                    f"{path}/r*i1p2f1/{realm}/{v}/gn/v20190429/{v}_{realm}_CanESM5_historical_r*i1p2f1_gn_185001-201412.nc"
                )
            )
            d0 = xr.open_dataset(
                files[0],
                chunks={},
            )[v]

            delayed = dask.array.stack(
                [_open_CanESM5_hist(f, v) for f in files], axis=0
            )

            ds.append(
                xr.DataArray(
                    delayed,
                    dims=["member", *d0.dims],
                    coords={
                        "member": members,
                        **d0.coords,
                    },
                    attrs=d0.attrs,
                ).to_dataset(name=v)
            )

        return xr.merge(ds).compute()