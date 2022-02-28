from pathlib import Path

import logging

import xarray as xr

from src import utils

from climpred import HindcastEnsemble


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data/processed"


def verify(config, save_dir, save=True):
    """
    Prepare a skill metric according to a provided config file

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

    if "prepare" in cfg:
        prepared = []

        for identifier, params in cfg["prepare"].items():
            hcst = xr.open_zarr(f"{DATA_DIR}/{params['hindcasts']}.zarr").unify_chunks()
            hcst = hcst.drop("time")  # time will be added by climpred
            hindcast = HindcastEnsemble(hcst)

            obsv = xr.open_zarr(
                f"{DATA_DIR}/{params['observations']}.zarr"
            ).unify_chunks()
            hindcast = hindcast.add_observations(obsv)

            if "simulations" in params:
                hist = xr.open_zarr(
                    f"{DATA_DIR}/{params['simulations']}.zarr"
                ).unify_chunks()
                hindcast = hindcast.add_uninitialized(hist)

            ds = hindcast.verify(**params["verify"])
            prepared.append(ds)
            if save:
                ds = ds.chunk("auto").unify_chunks()
                for var in ds.variables:
                    ds[var].encoding = {}
                ds.to_zarr(f"{save_dir}/{identifier}.zarr", mode="w")

        return prepared
    else:
        raise ValueError(f"No skill metrics were specified to prepare")
