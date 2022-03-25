# Some functions for producing plots in the notebooks. Putting here to keep notebooks nice and clean

import functools

import numpy as np

import xarray as xr
 
import matplotlib.pyplot as plt

from src import plot


def plot_hindcasts(hindcasts, historicals, observations, timescale, variable, region=None):
    """
    Helper function for plotting hindcast timeseries. If the data are spatial, plots the global mean
    """

    def _load_dataset(dataset, timescale, variable, region):
        """Load a skill metric"""
        import os

        DATA_DIR = "../data/processed/"
        if "CAFE" in dataset:
            anom_period = "1991-2020"
        else:
            anom_period = "1985-2014"
        if region is not None:
            region = f"_{region}"
        else: region = ""
        file = f"{DATA_DIR}/{dataset}.{timescale}.anom_{anom_period}.{variable}{region}.zarr"
        return xr.open_zarr(file)

    def global_mean(ds):
        return (
            ds.sel(lat=slice(-60, 60))
            .weighted(ds["area"])
            .mean(["lon", "lat"], keep_attrs=True)
        )

    if region == "Aus_NRM":
        fig = plt.figure(figsize=(5*len(hindcasts), 17.5))
        n_rows = 5
        n_columns = len(hindcasts)
        axs = np.array(fig.subplots(n_rows, n_columns)).T.flatten()
    else:
        fig = plt.figure(figsize=(15, 3.5*len(hindcasts)))
        n_rows = len(hindcasts)
        n_columns = 1
        axs = np.array(fig.subplots(n_rows, n_columns)).flatten()
    
    ax_n = 0
    for idx, hindcast in enumerate(hindcasts):
        hindcast_data = _load_dataset(hindcast, timescale, variable, region)
        if region == "global":
            hindcast_data = global_mean(hindcast_data)
        hindcast_dict = {hindcast: hindcast_data.compute()}
                         
        try:
            historical_data = _load_dataset(historicals[idx], timescale, variable, region)
            if region == "global":
                historical_data = global_mean(historical_data)
            historical_dict = {historicals[idx]: historical_data.compute()}
        except IndexError:
            historical_dict = None

        observations_dict = {}
        for observation in observations:
            observation_data = _load_dataset(observation, timescale, variable, region)
            if region == "global":
                observation_data = global_mean(observation_data)
            observations_dict[observation] = observation_data.compute()

        if region == "Aus_NRM":
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
    hindcast, reference, timescale, variable, metric, region, verif_period=None
):
    """Load a skill metric"""
    SKILL_DIR = "../data/skill/"
    if hindcast == "CAFEf6":
        anom_period = "1991-2020"
    else:
        anom_period = "1985-2014"
    if verif_period is None:
        verif_period = anom_period
    if region is not None:
        region = f"_{region}"
    else:
        region = ""
    file = (
        f"{SKILL_DIR}/{hindcast}.{reference}.{timescale}.anom_{anom_period}"
        f".{variable}{region}.{metric}_{verif_period}.zarr"
    )
    return xr.open_zarr(file).compute()


def plot_metrics(
    hindcasts,
    reference,
    timescales,
    variable,
    metrics,
    region=None,
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
    verif_period=None,
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
                    region="global",
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
                    region="global",
                    verif_period=verif_period,
                )
            )

        if len(metrics) == 1:
            annual = annual_metrics[0]
            quadrennial = quadrennial_metrics[0]
            add_colorbar = True
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
                return (1 * positive + xr.where(positive & significant, 1, 0)) / 2

            annual = functools.reduce(
                both_pos_and_signif, [pos_and_signif(ds) for ds in annual_metrics]
            )
            quadrennial = functools.reduce(
                both_pos_and_signif, [pos_and_signif(ds) for ds in quadrennial_metrics]
            )
            add_colorbar = False

        # Change this to change what leads are plotted
        to_plot = {
            "year 1": annual.sel(lead=23),
            "years 1-4": quadrennial.sel(lead=59),
            "years 5-8": quadrennial.sel(lead=107),
        }
        fields.append(list(to_plot.values()))
        headings.append(
            [f"{hindcast} | {metric} | {timescale}" for timescale in to_plot.keys()]
        )

    if len(hindcasts) >= 3:
        figsize = (15, 9.1)
    elif len(hindcasts) == 2:
        figsize = (15, 6.1)
    else:
        figsize = (15, 3.2)
    return plot.metric_maps(
        fields,
        variable=variable,
        vrange=(-1, 1),
        headings=headings,
        add_colorbar=add_colorbar,
        figsize=figsize,
    )