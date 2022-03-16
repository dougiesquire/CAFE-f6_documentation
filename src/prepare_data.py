import os
import glob
import tempfile
from pathlib import Path

import logging
import argparse

import dask
from dask.distributed import Client

import xarray as xr

from src import utils


dask.config.set(**{"array.slicing.split_large_chunks": False})
xr.set_options(keep_attrs=True)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data/raw"


# Dataset-specific opening code
# ===============================================


class _open:
    """
    Class containing the dataset-specific code for opening each available dataset
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def GPCP(variables, _, preprocess):
        """Open GPCP v2.3 variables"""
        files = sorted(glob.glob(f"{DATA_DIR}/GPCP/????/*.nc"))
        ds = xr.open_mfdataset(
            files,
            parallel=True,
            use_cftime=True,
        )[variables]
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def CAFE60v1(variables, realm, preprocess):
        """Open CAFE60v1 variables from specified realm"""
        file = DATA_DIR / f"CAFE60v1/{realm}.zarr.zip"
        ds = xr.open_dataset(file, engine="zarr", chunks={})[variables]
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    @staticmethod
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

    @staticmethod
    def _cmip6_dcppA_hindcast(
        model, variant_id, grid, variables, realm, years, members, version
    ):
        """Open CMIP6 dcppA-hindcast variables from specified monthly realm"""

        def _dcpp_file(y, m, v):
            path = f"{DATA_DIR}/{model}/s{y}-r{m}{variant_id}/{realm}/{v}/{grid}"
            if version == "latest":
                versions = sorted(glob.glob(f"{path}/v????????"))
                if len(versions) == 0:
                    raise ValueError(f"No versions found for {path}")
                else:
                    path = versions[-1]
            else:
                path = f"{path}/{version}"

            file_pattern = (
                f"{v}_{realm}_{model}_dcppA-hindcast_s{y}-r{m}{variant_id}_{grid}_*.nc"
            )
            files = sorted(glob.glob(f"{path}/{file_pattern}"))
            if len(files) == 0:
                raise ValueError(f"No files found for {path}/{file_pattern}")
            else:
                return files

        @dask.delayed
        def _open_dcpp_delayed(y, m, v):
            files = _dcpp_file(y, m, v)
            return xr.concat(
                [xr.open_dataset(f, chunks={}, use_cftime=True) for f in files],
                dim="time",
            )[v]

        def _open_dcpp(y, m, v, d0):
            var_data = _open_dcpp_delayed(y, m, v).data
            return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

        ds = []
        for v in variables:
            f0 = _dcpp_file(years[0], members[0], v)
            ds0 = xr.concat(
                [xr.open_dataset(f, chunks={}, use_cftime=True) for f in f0], dim="time"
            )
            ds0 = utils.convert_time_to_lead(ds0, time_freq="months")
            ds0 = utils.round_to_start_of_month(ds0, dim="init")
            d0 = ds0[v]

            delayed = []
            for y in years:
                delayed.append(
                    dask.array.stack([_open_dcpp(y, m, v, d0) for m in members], axis=0)
                )
            delayed = dask.array.stack(delayed, axis=0)

            init = xr.cftime_range(
                ds0.init.dt.strftime(
                    "%Y-%m-%d"
                ).item(),  # Already rounded to start of month
                periods=len(years),
                freq="12MS",
                calendar="julian",
            )
            time = [
                xr.cftime_range(
                    i, periods=ds0.sizes["lead"], freq="MS", calendar="julian"
                )
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

    @staticmethod
    def CanESM5(variables, realm, preprocess):
        """Open CanESM5 dcppA-hindcast variables from specified monthly realm"""
        model = "CanESM5"
        variant_id = "i1p2f1"
        grid = "gn"
        years = range(1960, 2016 + 1)
        members = range(1, 40 + 1)
        version = "v20190429"
        ds = _open._cmip6_dcppA_hindcast(
            model, variant_id, grid, variables, realm, years, members, version
        )
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    @staticmethod
    def EC_Earth3(variables, realm, preprocess):
        """Open EC-Earth3 dcppA-hindcast variables from specified monthly realm"""
        model = "EC-Earth3"
        variant_id = "i1p1f1"
        grid = "gr"
        years = range(1960, 2018 + 1)
        members = range(1, 10 + 1)
        version = "v2020121?"
        ds = _open._cmip6_dcppA_hindcast(
            model, variant_id, grid, variables, realm, years, members, version
        )
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    #     def CESM1(variables, realm, preprocess):
    #         """
    #         NOT CURRENTLY USED
    #         Open CESM1-1-CAM5-CMIP5 dcppA-hindcast variables from specified monthly realm
    #         """
    #         model = "CESM1-1-CAM5-CMIP5"
    #         variant_id = "i1p1f1"
    #         grid = "gr" if realm == "Omon" else "gn"
    #         years = range(1981, 2017 + 1)  # CESM1 ocean files end in 2017
    #         members = range(1, 40 + 1)
    #         version = "v20191016" if realm == "Omon" else "v20191007"
    #         ds = _open._cmip6_dcppA_hindcast(
    #             model,
    #             variant_id,
    #             grid,
    #             variables,
    #             realm,
    #             years,
    #             members,
    #             version)
    #         if preprocess is not None:
    #             return preprocess(ds)
    #         else:
    #             return ds

    @staticmethod
    def _cmip6_historical(model, variant_id, grid, variables, realm, members, version):
        """
        Open CMIP6 historical variables from specified realm

        Can specify version='latest' but this is slower as it has to search each
        directory for the latest version
        """

        def _hist_files(m, v):
            path = f"{DATA_DIR}/{model}_hist/r{m}{variant_id}/{realm}/{v}/{grid}"
            if version == "latest":
                versions = sorted(glob.glob(f"{path}/v????????"))
                if len(versions) == 0:
                    raise ValueError(f"No versions found for {path}")
                else:
                    path = versions[-1]
            else:
                path = f"{path}/{version}"
            file_pattern = f"{v}_{realm}_{model}_historical_r*{variant_id}_{grid}_*.nc"
            files = sorted(glob.glob(f"{path}/{file_pattern}"))
            if len(files) == 0:
                raise ValueError(f"No files found for {path}/{file_pattern}")
            else:
                return files

        @dask.delayed
        def _open_hist_delayed(m, v):
            files = _hist_files(m, v)
            return xr.concat(
                [xr.open_dataset(f, chunks={}, use_cftime=True) for f in files],
                dim="time",
            )[v]

        def _open_hist(m, v):
            var_data = _open_hist_delayed(m, v).data
            return dask.array.from_delayed(var_data, d0.shape, d0.dtype)

        ds = []
        for v in variables:
            f0 = _hist_files(members[0], v)
            d0 = xr.concat(
                [xr.open_dataset(f, chunks={}, use_cftime=True) for f in f0], dim="time"
            )[v]

            delayed = dask.array.stack([_open_hist(m, v) for m in members], axis=0)

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

    @staticmethod
    def CanESM5_hist(variables, realm, preprocess):
        """Open CanESM5 historical variables from specified realm"""
        model = "CanESM5"
        variant_id = "i1p2f1"
        grid = "gn"
        members = range(1, 40 + 1)
        version = "v20190429"
        ds = _open._cmip6_historical(
            model, variant_id, grid, variables, realm, members, version
        )
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds

    @staticmethod
    def EC_Earth3_hist(variables, realm, preprocess):
        """Open EC-Earth3 historical variables from specified realm"""
        model = "EC-Earth3"
        variant_id = "i1p1f1"
        grid = "gn" if realm == "Omon" else "gr"
        # Member 3 has screwy lats that can't be readily concatenated
        # Members 11, 13, 15 start in 1849
        # Member 19 has very few variables replicated for Omon
        # Members 101-150 only span 197001-201412
        members = [1, 2, 4, 6, 7, 9, 10, 12, 14, 16]  # , 17, 18] + list(range(20, 26))
        version = "latest"
        ds = _open._cmip6_historical(
            model, variant_id, grid, variables, realm, members, version
        )
        if preprocess is not None:
            return preprocess(ds)
        else:
            return ds


# Preparation
# ===============================================


def maybe_generate_CAFE_grid_files():
    """Generate files containing CAFE grids"""
    path = DATA_DIR / "CAFE_hist/"

    atmos_file = DATA_DIR / "gridinfo/CAFE_atmos_grid.nc"
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

    ocean_file = DATA_DIR / "gridinfo/CAFE_ocean_grid.nc"
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


# def maybe_generate_HadISST_grid_file():
#     """Generate file containing HadISST grid"""
#     file = DATA_DIR / "gridinfo/HadISST_grid.nc"
#     if not os.path.exists(file):
#         path = DATA_DIR / "HadISST/ocean_month.zarr"
#         had = xr.open_zarr(path)[["sst"]].isel(time=0).drop("time")
#         grid = xr.zeros_like(
#             had.rename({"sst": "HadISST_grid", "latitude": "lat", "longitude": "lon"})
#         )
#         grid.attrs = {}
#         grid.to_netcdf(file, mode="w")


# Command line interface
# ===============================================


def prepare_dataset(config, save_dir, save=True):
    """
    Prepare a dataset according to a provided config file

    Parameters
    ----------
    config : str
        The name of the config file
    save_dir : str
        The directory to save to
    save : boolean, optional
        If True (default), save the prepared dataset(s) in zarr format to save_dir. If
        False, return an xarray Dataset containing the prepared data. The latter is
        useful for debugging
    """
    logger = logging.getLogger(__name__)

    cfg = utils.load_config(config)

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
        prepared = []

        for identifier, params in cfg["prepare"].items():
            input_variables = params["uses"]
            if isinstance(input_variables, list):
                input_variables = {None: input_variables}

            # Build composite preprocess function
            if "preprocess" in params:
                preprocess = utils.composite_function(params["preprocess"])
            else:
                preprocess = None

            logger.info(f"Processing {identifier} from {cfg['name']}")
            ds = []
            for realm, var in input_variables.items():
                if realm == "prepared":
                    ds.append(
                        xr.merge(
                            xr.open_zarr(f"{save_dir}/{cfg['name']}.{v}.zarr")
                            for v in var
                        )
                    )
                else:
                    if hasattr(_open, cfg["name"]):
                        ds.append(getattr(_open, cfg["name"])(var, realm, preprocess))
                    else:
                        raise ValueError(
                            f"There is no method available to open '{cfg['name']}'. Please ensure that the 'name' entry in the config file matches an existing method in src.data._open, or add a new method for this data. Available methods are {methods}"
                        )
            ds = xr.merge(ds)

            if "apply" in params:
                ds = utils.composite_function(params["apply"])(ds)

            prepared.append(ds)
            if save:
                ds = ds.unify_chunks()
                for var in ds.variables:
                    ds[var].encoding = {}
                ds.to_zarr(f"{save_dir}/{cfg['name']}.{identifier}.zarr", mode="w")

        return prepared

    else:
        raise ValueError(f"No variables were specified to prepare")


def main(config, config_dir, save_dir):
    """
    Spin up a dask cluster and process and save raw data according to a provided config file

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
        default=f"{PROJECT_DIR}/config/prepare_data",
        help="Location of directory containing config file(s) to use, defaults to <project_dir>/config/prepare_data",
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
