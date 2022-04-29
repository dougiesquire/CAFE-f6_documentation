# Some functions for producing plots in the notebooks. Putting here to keep notebooks nice and clean

from pathlib import Path

import functools

import numpy as np

import xarray as xr
 
import matplotlib.pyplot as plt

from src import plot, utils

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

def plot_hindcasts(hindcasts, historicals, observations, timescale, variable, region=None, diagnostic="anom"):
    """
    Helper function for plotting hindcast timeseries. If the data are spatial, plots the global mean
    """

    def _load_dataset(dataset, timescale, variable, region, diagnostic):
        """Load a skill metric"""
        
        if region is not None:
            region = f"_{region}"
        else: region = ""
        
        try:
            file1 = f"{DATA_DIR}/processed/{dataset}.{timescale}.{diagnostic}.{variable}{region}.zarr"
            ds = xr.open_zarr(file1, consolidated=True, decode_timedelta=False)
        except:
            try:
                file2 = f"{DATA_DIR}/processed/{dataset}.{timescale}.{diagnostic}_{train_period}.{variable}{region}.zarr"
                ds = xr.open_zarr(file2, consolidated=True, decode_timedelta=False)
            except:
                raise OSError(f"Could not find {file1} or {file2}")
        
        if region == "_Aus":
            # Mask out land
            shapefile = f"{DATA_DIR}/raw/NRM_super_clusters/NRM_super_clusters.shp"
            mask = utils.get_region_masks_from_shp(
                ds, shapefile, "label"
            ).sum("region")
            ds = ds.where(mask)
        
        return ds

    def area_mean(ds):
        return (
            ds.sel(lat=slice(-60, 60))
            .weighted(ds["area"])
            .mean(["lon", "lat"], keep_attrs=True)
        )
    
    if region == "Aus_NRM":
        regional = True
    else:
        regional = False

    if regional:
        fig = plt.figure(figsize=(15, 17.5))
        n_rows = 5
        n_columns = len(hindcasts)
        axs = np.array(fig.subplots(n_rows, n_columns)).T.flatten()
    else:
        fig = plt.figure(figsize=(15, 4*len(hindcasts)))
        n_rows = len(hindcasts)
        n_columns = 1
        axs = np.array(fig.subplots(n_rows, n_columns)).flatten()
        
    ax_n = 0
    for idx, hindcast in enumerate(hindcasts):
        if "CAFE" in hindcast:
            train_period = "1991-2020"
        else:
            train_period = "1985-2014"
                
        hindcast_data = _load_dataset(hindcast, timescale, variable, region, diagnostic)
        if (not regional) & (region is not None):
            hindcast_data = area_mean(hindcast_data)
        hindcast_dict = {hindcast: hindcast_data.compute()}
                         
        try:
            historical_data = _load_dataset(historicals[idx], timescale, variable, region, diagnostic)
            if (not regional) & (region is not None):
                historical_data = area_mean(historical_data)
            historical_dict = {historicals[idx]: historical_data.compute()}
        except IndexError:
            historical_dict = None

        observations_dict = {}
        for observation in observations:
            observation_data = _load_dataset(observation, timescale, variable, region, diagnostic)
            if (not regional) & (region is not None):
                observation_data = area_mean(observation_data)
            observations_dict[observation] = observation_data.compute()

        if regional:
            for reg in hindcast_dict[list(hindcast_dict.keys())[0]].region:
                hindcast_dict_region = {k: v.sel(region=reg) for k,v in hindcast_dict.items()}
                historical_dict_region = {k: v.sel(region=reg) for k,v in historical_dict.items()}
                observations_dict_region = {k: v.sel(region=reg) for k,v in observations_dict.items()}
                ax = plot.hindcasts(hindcast_dict_region, observations_dict_region, historical_dict_region, ax=axs[ax_n])
                ax.set_title(reg.item())
                ax_n += 1
        else:
            _ = plot.hindcasts(hindcast_dict, observations_dict, historical_dict, ax=axs[ax_n])
            ax_n += 1
        
        
def _load_skill_metric(
    hindcast, reference, timescale, variable, metric, region, diagnostic, verif_period=None
):
    """Load a skill metric"""

    if verif_period is None:
        if hindcast == "CAFEf6":
            verif_period = "1991-2020"
        else:
            verif_period = "1985-2014"
            
    if region is not None:
        region = f"_{region}"
    else:
        region = ""
        
    try:
        file1 = (
            f"{DATA_DIR}/skill/{hindcast}.{reference}.{timescale}.{diagnostic}"
            f".{variable}{region}.{metric}_{verif_period}.zarr"
        )
        ds = xr.open_zarr(file1, consolidated=True, decode_timedelta=False).compute()
    except:
        if hindcast == "CAFEf6":
            train_period = "1991-2020"
        else:
            train_period = "1985-2014"
            
        try:
            file2 = (
                f"{DATA_DIR}/skill/{hindcast}.{reference}.{timescale}.{diagnostic}_{train_period}"
                f".{variable}{region}.{metric}_{verif_period}.zarr"
            )
            ds = xr.open_zarr(file2, consolidated=True, decode_timedelta=False).compute()
        except:
            raise OSError(f"Could not find {file1} or {file2}")
            
    if region == "_Aus":
        # Mask out land
        shapefile = f"{DATA_DIR}/raw/NRM_super_clusters/NRM_super_clusters.shp"
        mask = utils.get_region_masks_from_shp(
            ds, shapefile, "label"
        ).sum("region").assign_coords({"region": "Australia"})
        ds = ds.where(mask)
            
    return ds


def plot_metrics(
    hindcasts,
    reference,
    timescales,
    variable,
    metrics,
    region=None,
    diagnostic="anom",
    verif_period=None,
):
    """
    Helper function for plotting some skill metrics.
    """
    
    if region == "Aus_NRM":
        n_regions = 5
        selections = [{"region": i} for i in range(5)]
        figsize = (9, 12)
    else:
        selections = [None]
        figsize = (9, 4)

    panels = []
    headings = []
    for sel in selections:
        row_list = []
        for timescale in timescales:
            metric_dict = {}
            for hindcast in hindcasts:
                model_metrics = {}
                for metric in metrics:
                    try:
                        model_metric = _load_skill_metric(
                            hindcast,
                            reference,
                            timescale,
                            f"{variable}",
                            metric,
                            region,
                            diagnostic,
                            verif_period,
                        ).isel(sel)
                        model_metrics[metric] = model_metric
                        if sel is not None:
                            region_name = (
                                f"{model_metric[list(sel.keys())[0]].item()}, "
                            )
                        else:
                            region_name = ""
                    except:
                        pass
                metric_dict[hindcast] = model_metrics
            row_list.append(metric_dict)
        panels.append(row_list)
        headings.append([f"{region_name}{timescale}" for timescale in timescales])
    return plot.metrics(panels, variable, headings, figsize=figsize)


def plot_metric_maps(
    hindcasts,
    reference,
    variable,
    metrics,
    region="global",
    diagnostic="anom",
    verif_period=None,
    vrange=(-1,1),
    add_colorbar=True,
    cbar_bounds=None,
    cmap="PiYG",
    central_longitude=180,
    figsize=None
):
    """
    Helper function for plotting some skill maps. Edit this function to change which
    lead times are plotted.

    When multiple metrics are provided the regions where all metrics are positive and
    significant are plotted.
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    fields = []
    headings = []
    for hindcast in hindcasts:
        annual_metrics = []
        quadrennial_metrics = []
        for metric in metrics:
            annual_metrics.append(
                _load_skill_metric(
                    hindcast,
                    reference,
                    "annual",
                    f"{variable}",
                    metric,
                    region=region,
                    diagnostic=diagnostic,
                    verif_period=verif_period,
                )
            )
            quadrennial_metrics.append(
                _load_skill_metric(
                    hindcast,
                    reference,
                    "4-year",
                    f"{variable}",
                    metric,
                    region=region,
                    diagnostic=diagnostic,
                    verif_period=verif_period,
                )
            )

        if len(metrics) == 1:
            annual = annual_metrics[0]
            quadrennial = quadrennial_metrics[0]
        else:

            def both_pos_and_signif(ds_1, ds_2):
                """Return array where two arrays are both positive and signif"""
                positive = (ds_1 > 0) & (ds_2 > 0)
                significant = (ds_1 == 1) & (ds_2 == 1)
                positive_and_significant = (1 * positive + 1 * significant) / 2
                return positive_and_significant.where(positive_and_significant > 0)

            def pos_and_signif(ds):
                """How to select regions when multiple metrics are provided"""
                positive = ds[[variable]] > 0
                significant = ds[f"{variable}_signif"]
                # Deal with nans
                significant = xr.where(significant.notnull(), significant.astype(bool), False)
                return ((1*positive) + xr.where(positive & significant, 1, 0)) / 2

            annual = functools.reduce(
                both_pos_and_signif, [pos_and_signif(ds) for ds in annual_metrics]
            )
            quadrennial = functools.reduce(
                both_pos_and_signif, [pos_and_signif(ds) for ds in quadrennial_metrics]
            )
            add_colorbar = False

        # Change this to change what leads are plotted
        to_plot = {
            "year 1": annual.isel(lead=1), #.sel(lead=23),
            "years 1-4": quadrennial.isel(lead=4), #.sel(lead=59),
            "years 5-8": quadrennial.isel(lead=8), #.sel(lead=107),
        }
        fields.append(list(to_plot.values()))
        headings.append(
            [f"{hindcast} | {metric} | {timescale}" for timescale in to_plot.keys()]
        )

    if figsize is None:
        if len(hindcasts) >= 3:
            figsize = (15, 9.1)
        elif len(hindcasts) == 2:
            figsize = (15, 6.1)
        else:
            figsize = (15, 3.2)
    return plot.metric_maps(
        fields,
        variable=variable,
        vrange=vrange,
        headings=headings,
        add_colorbar=add_colorbar,
        cbar_bounds=cbar_bounds,
        cmap=cmap,
        central_longitude=central_longitude,
        figsize=figsize,
    )