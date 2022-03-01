from pathlib import Path

import logging
import argparse

import xarray as xr

from src import utils

import climpred
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
            if "apply" in params:
                if "all" in params["apply"]:
                    params["apply"]["hindcasts"] = params["apply"]["all"]
                    params["apply"]["observations"] = params["apply"]["all"]
                    if "simulations" in params:
                        params["apply"]["simulations"] = params["apply"]["all"]
            else: params["apply"] = []
            
            hcst = xr.open_zarr(f"{DATA_DIR}/{params['hindcasts']}.zarr").unify_chunks()
            hcst = hcst.drop("time")  # time will be added by climpred
            if "hindcasts" in params["apply"]:
                hcst = utils.composite_function(params["apply"]["hindcasts"])(hcst)
            hindcast = HindcastEnsemble(hcst)

            obsv = xr.open_zarr(
                f"{DATA_DIR}/{params['observations']}.zarr"
            ).unify_chunks()
            if "observations" in params["apply"]:
                obsv = utils.composite_function(params["apply"]["observations"])(obsv)
            hindcast = hindcast.add_observations(obsv)

            if "simulations" in params:
                hist = xr.open_zarr(
                    f"{DATA_DIR}/{params['simulations']}.zarr"
                ).unify_chunks()
                if "simulations" in params["apply"]:
                    hist = utils.composite_function(params["apply"]["simulations"])(hist)
                hindcast = hindcast.add_uninitialized(hist)
                    
            ds = hindcast.verify(**params["verify"])
            
            # Add the verification period to the attributes if convenient
            if "alignment" in params["verify"]:
                if params["verify"]["alignment"] == "same_verifs":
                    verif_dates = climpred.alignment.return_inits_and_verif_dates(
                        hcst.rename({"init": "time"}),
                        obsv, 
                        alignment=params["verify"]["alignment"])[1]
                    # Confirm all leads are have the same verif dates
                    verif_dates = [v for v in verif_dates.values()]
                    verif_dates_0 = verif_dates[0]
                    assert all([all(v == verif_dates_0) for v in verif_dates])
                    ds.attrs["verification period start"] = f"{verif_dates_0[0]}"
                    ds.attrs["verification period end"] = f"{verif_dates_0[-1]}"
                             
            prepared.append(ds)
            if save:
                ds = ds.chunk("auto").unify_chunks()
                for var in ds.variables:
                    ds[var].encoding = {}
                ds.to_zarr(f"{save_dir}/{identifier}.zarr", mode="w")

        return prepared
    else:
        raise ValueError(f"No skill metrics were specified to prepare")


def main(config, config_dir, save_dir):
    """
    Prepare skill metrics according to a provided config file

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

    logger.info(f"Preparing skill metrics using {config}")
    verify(f"{config_dir}/{config}", save_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(
        description="Prepare skill metrics according to a provided config file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Configuration file to process",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=f"{PROJECT_DIR}/config/verify",
        help="Location of directory containing config file(s) to use, defaults to <project_dir>/config/verify",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=f"{PROJECT_DIR}/data/skill/",
        help="Location of directory to save skill data to, defaults to <project_dir>/data/skill/",
    )

    args = parser.parse_args()
    config = args.config
    config_dir = args.config_dir
    save_dir = args.save_dir

    main(config, config_dir, save_dir)