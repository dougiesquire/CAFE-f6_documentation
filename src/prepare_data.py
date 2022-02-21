import os
import glob
import tempfile
from pathlib import Path

import yaml

import logging
import argparse

import dask
from dask.distributed import Client

import xarray as xr

from functools import reduce, partial

from src import utils


dask.config.set(**{"array.slicing.split_large_chunks": False})


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data/raw"


def _load_config(name):
    """Load a config .yml file for a specified dataset"""
    with open(name, "r") as reader:
        return yaml.load(reader, Loader=yaml.SafeLoader)


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
            except KeyError:
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


def _scale_variables(ds, norm_dict):
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
        step of a config .yml
    """

    def composite(*funcs):
        def compose(f, g):
            return lambda x: g(f(x))

        return reduce(compose, funcs, lambda x: x)

    funcs = []
    for fn in function_dict.keys():
        kws = function_dict[fn]
        kws = {} if kws is None else kws
        funcs.append(partial(getattr(utils, fn), **kws))

    return composite(*funcs)


class _open:
    """
    Class containing the dataset-specific code for opening each available dataset
    """

    def JRA55(variables, realm, preprocess):
        """Open JRA55 variables from specified realm"""
        file = DATA_DIR / f"JRA55/{realm}.zarr.zip"
        ds = xr.open_dataset(
            file,
            engine="zarr",
            chunks={},
            use_cftime=True,
        )[variables]
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def HadISST(variables, realm, preprocess):
        """Open HadISST variables from specified realm"""
        file = DATA_DIR / f"HadISST/{realm}.zarr"
        ds = xr.open_dataset(
            file,
            engine="zarr",
            chunks={},
            use_cftime=True,
        )[variables]
        ds = ds.where(ds > -1000)
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def EN422(variables, _, preprocess):
        """Open EN.4.2.2 variables"""
        files = sorted(glob.glob(f"{DATA_DIR}/EN.4.2.2/*.nc"))
        ds = xr.open_mfdataset(
            files,
            parallel=True,
            use_cftime=True,
        )[variables]
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def CAFEf6(variables, realm, preprocess):
        """Open CAFE-f6 variables from specified realm applying preprocess prior to
        concanenating forecasts
        """
        files = sorted(
            glob.glob(f"{DATA_DIR}/CAFEf6/c5-d60-pX-f6-????1101/{realm}.zarr.zip")
        )
        return xr.open_mfdataset(
            files,
            compat="override",
            preprocess=preprocess,
            engine="zarr",
            coords="minimal",
            parallel=True,
        )[variables]

    def CAFEf5(variables, realm, preprocess):
        """Open CAFE-f5 variables from specified realm, including appending first
        10 members of CAFE-f6 for 2020 forecast
        """
        file = DATA_DIR / f"CAFEf5/NOV/{realm}.zarr.zip"
        ds = xr.open_dataset(file, engine="zarr", chunks={})[variables]
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def CAFE60v1(variables, realm, preprocess):
        """Open CAFE60v1 variables from specified realm"""
        file = DATA_DIR / f"CAFE60v1/{realm}.zarr.zip"
        ds = xr.open_dataset(file, engine="zarr", chunks={})[variables]
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def CAFE_hist(variables, realm, preprocess):
        """Open CAFE historical run variables from specified realm"""
        hist_file = DATA_DIR / f"CAFE_hist/{realm}.zarr.zip"
        hist = xr.open_dataset(hist_file, engine="zarr", chunks={})[variables]

        ctrl_file = DATA_DIR / f"CAFE_ctrl/{realm}.zarr.zip"
        ctrl = xr.open_dataset(ctrl_file, engine="zarr", chunks={})[variables]

        hist = utils.truncate_latitudes(hist)
        ctrl = utils.truncate_latitudes(ctrl)

        drift = (
            ctrl.mean("ensemble")
            .groupby("time.month")
            .map(lambda x: x - x.mean(["time"]))
        )
        ds = hist - drift
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def _dcppA_hindcast(model, variant_id, grid, variables, realm, years, members, version="*"):
        """Open dcppA-hindcast variables from specified monthly realm"""

        def _dcpp_file(y, m, v):
            pattern = f"{DATA_DIR}/{model}/s{y}-r{m}{variant_id}/{realm}/{v}/{grid}/{version}/{v}_{realm}_{model}_dcppA-hindcast_s{y}-r{m}{variant_id}_{grid}_*.nc"
            file = sorted(glob.glob(pattern))
            if len(file) == 0:
                raise ValueError(f"No files found for {pattern}")
            else:
                return file

        @dask.delayed
        def _open_dcpp_delayed(y, m, v):
            files = _dcpp_file(y, m, v)
            ds = xr.concat(
                [xr.open_dataset(f, chunks={}, use_cftime=True) for f in files],
                dim="time"
            )[v]
            return ds

        def _open_dcpp(y, m, v, d0):
            var_data = _open_dcpp_delayed(y, m, v).data
            return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

        ds = []
        for v in variables:
            f0 = _dcpp_file(years[0], members[0], v)
            ds0 = xr.open_mfdataset(f0, chunks={}, use_cftime=True, parallel=False)
            ds0 = xr.concat(
                [xr.open_dataset(f, chunks={}, use_cftime=True) for f in f0],
                dim = "time"
            )
            ds0 = utils.convert_time_to_lead(ds0)
            ds0 = utils.round_to_start_of_month(ds0, dim="init")  
            d0 = ds0[v]
            
            delayed = []
            for y in years:
                delayed.append(
                    dask.array.stack(
                        [_open_dcpp(y, m, v, d0) for m in members], axis=0
                    )
                )
            delayed = dask.array.stack(delayed, axis=0)

            init = xr.cftime_range(
                ds0.init.dt.strftime("%Y-%m-%d").item(), # Already rounded to start of month
                periods=len(years), 
                freq="12MS", 
                calendar="julian"
            )
            time = [
                xr.cftime_range(i, periods=ds0.sizes["lead"], freq="MS", calendar="julian")
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
        
    def CanESM5(variables, realm, preprocess):
        """Open CanESM5 dcppA-hindcast variables from specified monthly realm"""
        model = "CanESM5"
        variant_id = "i1p2f1"
        grid = "gn"
        years = range(1980, 2016 + 1)
        members = range(1, 40 + 1)
        version = "v20190429"
        ds = _open._dcppA_hindcast(
            model, 
            variant_id, 
            grid,
            variables, 
            realm, 
            years, 
            members, 
            version)
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds
        
    def CESM1(variables, realm, preprocess):
        """Open CESM1-1-CAM5-CMIP5 dcppA-hindcast variables from specified monthly realm"""
        model = "CESM1-1-CAM5-CMIP5"
        variant_id = "i1p1f1"
        grid = "gr" if realm == "Omon" else "gn"
        years = range(1981, 2017 + 1)  # CESM1 ocean files end in 2017
        members = range(1, 40 + 1)
        version = "v20191016" if realm == "Omon" else "v20191007"
        ds = _open._dcppA_hindcast(
            model, 
            variant_id, 
            grid,
            variables, 
            realm, 
            years, 
            members, 
            version)
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds
        
    def EC_Earth3(variables, realm, preprocess):
        """Open EC-Earth3 dcppA-hindcast variables from specified monthly realm"""
        model = "EC-Earth3"
        variant_id = "i1p1f1"
        grid = "gr"
        years = range(1981, 2018 + 1)  # CESM1 ocean files end in 2017
        members = range(1, 10 + 1)
        version = "v2020121?"
        ds = _open._dcppA_hindcast(
            model, 
            variant_id, 
            grid,
            variables, 
            realm, 
            years, 
            members, 
            version)
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    def CanESM5_hist(variables, realm, preprocess):
        """Open CanESM5 historical variables from specified realm"""

        def _CanESM5_hist_files(v):
            version = "v20190429"
            return sorted(
                glob.glob(
                    f"{DATA_DIR}/CanESM5_hist/r*i1p2f1/{realm}/{v}/gn/{version}/{v}_{realm}_CanESM5_historical_r*i1p2f1_gn_185001-201412.nc"
                )
            )

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
            files = _CanESM5_hist_files(v)
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

        ds = xr.merge(ds).compute()
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds


def maybe_generate_CAFE_grid_files():
    """Generate files containing CAFE grids"""
    path = PROJECT_DIR / "data/raw/CAFE_hist/"

    atmos_file = PROJECT_DIR / "data/raw/gridinfo/CAFE_atmos_grid.nc"
    if not os.path.exists(atmos_file):
        atmos = (
            xr.open_zarr(f"{path}/atmos_hybrid_month.zarr.zip")
            .isel(time=0, ensemble=0)
            .drop(
                [
                    "time",
                    "ensemble",
                    "average_DT",
                    "average_T1",
                    "average_T2",
                ]
            )
        )
        atmos = utils.truncate_latitudes(atmos)
        atmos_grid = xr.zeros_like(
            atmos[["t_ref", "latb", "lonb"]].rename({"t_ref": "CAFE_atmos_grid"})
        )
        atmos_grid.attrs = {}
        atmos_grid.attrs = {}
        atmos_grid.to_netcdf(atmos_file, mode="w")

    ocean_file = PROJECT_DIR / "data/raw/gridinfo/CAFE_ocean_grid.nc"
    if not os.path.exists(ocean_file):
        ocean = (
            xr.open_zarr(f"{path}/ocean_month.zarr.zip")
            .isel(time=0, ensemble=0)
            .drop(
                [
                    "time",
                    "ensemble",
                    "average_DT",
                    "average_T1",
                    "average_T2",
                ]
            )
        )
        ocean_ut_grid = xr.zeros_like(
            ocean[["u", "area_t", "geolat_t", "geolon_t", "st_edges_ocean"]].rename(
                {"u": "CAFE_ocean_tu_grid"}
            )
        )
        ocean_tu_grid = xr.zeros_like(
            ocean[["wt", "area_t", "geolat_t", "geolon_t", "sw_edges_ocean"]].rename(
                {"wt": "CAFE_ocean_ut_grid"}
            )
        )
        ocean_grid = xr.merge([ocean_ut_grid, ocean_tu_grid])
        ocean_grid.attrs = {}
        ocean_grid.to_netcdf(ocean_file, mode="w")


def maybe_generate_HadISST_grid_file():
    """Generate file containing HadISST grid"""
    file = PROJECT_DIR / "data/raw/gridinfo/HadISST_grid.nc"
    if not os.path.exists(file):
        path = PROJECT_DIR / "data/raw/HadISST/ocean_month.zarr"
        had = xr.open_zarr(path)[["sst"]].isel(time=0).drop("time")
        grid = xr.zeros_like(
            had.rename({"sst": "HadISST_grid", "latitude": "lat", "longitude": "lon"})
        )
        grid.attrs = {}
        grid.to_netcdf(file, mode="w")


def prepare_dataset(config, save_dir):
    """
    Prepare a dataset according to a provided config file and save as zarr
    
    Parameters
    ----------
    config : str
        The name of the config file
    save_dir : str
        The directory to save to 
    """
    logger = logging.getLogger(__name__)

    cfg = _load_config(config)

    # List of datasets that have open methods impletemented
    methods = [
        method_name
        for method_name in dir(_open)
        if callable(getattr(_open, method_name))
    ]
    methods = [m for m in methods if "__" not in m]

    if "name" not in cfg:
        raise ValueError(
            f"Please provide an entry for 'name' in the config file so that I know how to open the data. Available options are {methods}"
        )

    if "prepare" in cfg:
        # Loop over output variables
        output_variables = cfg["prepare"]
        for variable in output_variables.keys():
            input_variables = output_variables[variable]["uses"]
            if isinstance(input_variables, list):
                input_variables = {None: input_variables}

            if "rename" in cfg:
                input_variables = _maybe_translate_variables(
                    input_variables, cfg["rename"]
                )

            if "preprocess" in output_variables[variable]:
                preprocess = _composite_function(
                    output_variables[variable]["preprocess"]
                )
            else:
                preprocess = None

            if hasattr(_open, cfg["name"]):
                ds = []
                logger.info(f"Processing {variable} from {cfg['name']}")
                for realm, var in input_variables.items():
                    ds.append(getattr(_open, cfg["name"])(var, realm, preprocess))
                ds = xr.merge(ds)
            else:
                raise ValueError(
                    f"There is no method available to open '{cfg['name']}'. Please ensure that the 'name' entry in the config file matches an existing method in src.data._open, or add a new method for this data. Available methods are {methods}"
                )

            if "rename" in cfg:
                ds = _maybe_rename(ds, cfg["rename"])

            if "scale_variables" in cfg:
                ds = _scale_variables(ds, cfg["scale_variables"])

            if "apply" in output_variables[variable]:
                ds = _composite_function(output_variables[variable]["apply"])(ds)

            for var in ds.data_vars:
                ds[var].encoding = {}
            ds.to_zarr(f"{save_dir}/{cfg['name']}.{variable}.zarr", mode="w")

    else:
        raise ValueError(f"No variables were specified to prepare")


def main(config, config_dir, save_dir):
    """
    Spin up a dask cluster and process raw data according to a provided config file

    Parameters
    ----------
    config : str
        The name of the config file
    config_dir : str
        The directory containing the config file
    save_dir : str
        The directory to save to
    """
    logger = logging.getLogger(__name__)

    logger.info("Spinning up a dask cluster")
    local_directory = tempfile.TemporaryDirectory()
    with Client(processes=False, local_directory=local_directory.name) as client:
        logger.info("Generating grid files")
        maybe_generate_HadISST_grid_file()
        maybe_generate_CAFE_grid_files()

        logger.info(f"Preparing raw data using {config}")
        prepare_dataset(f"{config_dir}/{config}", save_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(
        description="Process a raw dataset according to a provided config file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Configuration file to process",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=f"{PROJECT_DIR}/data/config/",
        help="Location of directory containing config file(s) to use, defaults to <project_dir>/data/config/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=f"{PROJECT_DIR}/data/processed/",
        help="Location of directory to save processed data to, defaults to <project_dir>/data/processed/",
    )

    args = parser.parse_args()
    config = args.config
    config_dir = args.config_dir
    save_dir = args.save_dir

    main(config, config_dir, save_dir)
