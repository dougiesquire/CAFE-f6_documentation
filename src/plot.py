import functools

import itertools
from itertools import cycle
from pathlib import Path

import cftime

import numpy as np
import xarray as xr

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import Divider, Size

import cartopy
import cartopy.crs as ccrs

PROJECT_DIR = Path(__file__).resolve().parents[1]

cartopy.config["pre_existing_data_dir"] = PROJECT_DIR / "data/cartopy-data"
cartopy.config["data_dir"] = PROJECT_DIR / "data/cartopy-data"


def hindcasts(hcsts, obsvs=None, hists=None, shade=False, ax=None, figsize=(15, 4)):
    """
    Plot sets of hindcasts. Where multiple variables are provided, it is
    assumed that all inputs contain the same variables.

    Parameters
    ----------
    hcsts : dict
        Dictionary of hindcasts to plot with the format {"name": hcst}, where
        hcst is an xarray.Dataset with dimensions "init" and "lead"
    obsvs : dict, optional
        Dictionary of observations to plot with the format {"name": obsv},
        where obsv is an xarray.Dataset with dimension "time"
    hist : dict, optional
        Dictionary of historical runs to plot with the format {"name": hist},
        where hist is an xarray.Dataset with dimension "time"
    shade : bool, optional
        If True, shade background according to change in bias correction in
        CAFE60v1
    """

    def _shading(ax):
        trans = cftime.datetime(1992, 1, 1)
        end = cftime.datetime(2040, 1, 1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between(
            [trans, end],
            [ylim[1], ylim[1]],
            [ylim[0], ylim[0]],
            color=[0.9, 0.9, 0.9],
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    n_vars = len(hcsts[list(hcsts.keys())[0]].data_vars)

    if ax is None:
        fig = plt.figure(figsize=(figsize[0], n_vars * figsize[1]))
        axs = fig.subplots(n_vars, 1, sharex=True)
        if n_vars == 1:
            axs = [axs]
    else:
        axs = [ax]

    # Plot the hindcasts
    colormaps = ["autumn", "winter", "cool"]
    colormapcycler = cycle(colormaps)
    for name, hcst in hcsts.items():
        if "time" in hcst:
            hcst_time = "time"
        elif "valid_time" in hcst:
            hcst_time = "valid_time"
        else:
            raise ValueError("I can't work out the time variable in hcsts")
        color = getattr(cm, next(colormapcycler))(np.linspace(0, 0.9, len(hcst.init)))
        for a, var in enumerate(hcst.data_vars):
            for idx, (i, c) in enumerate(zip(hcst[var].init, color)):
                if idx == 0:
                    label = name
                else:
                    label = "_nolabel_"

                h = hcst[var].sel(init=i)
                if "member" in h.dims:
                    h_mean = h.mean("member", keep_attrs=True)
                    for m in h.member:
                        axs[a].plot(
                            h[hcst_time],
                            h.sel(member=m),
                            color=[0.8, 0.8, 0.8],
                            linestyle="-",
                            label="_nolabel_",
                            zorder=-1,
                        )
                else:
                    h_mean = h
                axs[a].plot(
                    h_mean[hcst_time][0],
                    h_mean[0],
                    color=c,
                    marker="o",
                    label="_nolabel_",
                )
                axs[a].plot(
                    h_mean[hcst_time], h_mean, color=c, linestyle="-", label=label
                )
    xlim = (hcst[hcst_time].min().item(), hcst[hcst_time].max().item())

    # Plot the observations
    if obsvs is not None:
        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)
        for name, obsv in obsvs.items():
            line = next(linecycler)
            for a, var in enumerate(hcst.data_vars):
                axs[a].plot(
                    obsv.time, obsv[var], color="black", label=name, linestyle=line
                )

    # Plot the historical runs
    if hists is not None:
        for name, hist in hists.items():
            for a, var in enumerate(hist.data_vars):
                h_mean = (
                    hist[var].mean("member", keep_attrs=True)
                    if "member" in hist[var].dims
                    else hist[var]
                )
                axs[a].plot(h_mean.time, h_mean, label=name)

    # Format plots
    ticks = xr.cftime_range(start=xlim[0], end=xlim[-1], freq="5AS", calendar="julian")
    years = xr.cftime_range(start=xlim[0], end=xlim[-1], freq="AS", calendar="julian")
    xlim = (years.shift(-1, "AS")[0], years.shift(2, "AS")[-1])
    for a, var in enumerate(hcst.data_vars):
        axs[a].set_xticks(ticks.values)
        axs[a].set_xticklabels(ticks.year, rotation=40)
        axs[a].set_xlim(xlim)
        axs[a].set_ylabel(hcst[var].attrs["long_name"])
        axs[a].grid()
        if a == 0:
            axs[a].legend()
        if a == (n_vars - 1):
            axs[a].set_xlabel("year")
        else:
            axs[a].set_xlabel("")

        if shade:
            _shading(axs[a])

    plt.tight_layout()
    if ax is None:
        fig.patch.set_facecolor("w")
        return fig
    else:
        return ax


def metric_maps(
    fields, variable, vrange, headings=None, add_colorbar=True, figsize=(15, 15)
):
    """
    Plot panels of skill score maps

    Parameters
    ----------
    fields : list
        List of size n_rows x n_columns containing the fields to plot
    variable : str
        The name of the variable
    vrange : iterable of length 2
        The vmin and vmax values for all panels
    headings : list
        List of the same size as fields containing the headins for each panel
    figsize : iterable of length 2
        The total size of the figure
    """

    def _get_verif_period(ds):
        """Return a string of the verification period for a skill metric"""
        if "verification_time" in ds.coords:
            if "lead" in ds["verification_time"].dims:
                return "Lead-dependent verification period"
            else:
                return (
                    f"{ds['verification_time'].values[0].strftime('%Y')}-"
                    f"{ds['verification_time'].values[-1].strftime('%Y')}"
                )
        elif "verification_period" in ds.coords:
            if "lead" in ds["verification_period"].dims:
                return "Lead-dependent verification period"
            else:
                return (
                    f"{ds['verification_period'].item()[:4]}-"
                    f"{ds['verification_period'].item()[13:17]}"
                )
        elif "verification period start" in ds.coords:
            return (
                f"{ds.attrs['verification period start'][:4]}-"
                f"{ds.attrs['verification period end'][:4]}"
            )
        else:
            return "Unknown period"

    fig = plt.figure(figsize=figsize)
    n_rows = len(fields)
    n_columns = len(fields[0])
    axs = fig.subplots(
        n_rows,
        n_columns,
        sharex=True,
        sharey=True,
        subplot_kw=dict(projection=ccrs.PlateCarree(180)),
    )
    if n_rows == 1:
        axs = [axs]
        if n_columns == 1:
            axs = [axs]
    elif n_columns == 1:
        axs = [[ax] for ax in axs]

    cmap = cm.get_cmap("PiYG", 12)

    for r, c in itertools.product(range(n_rows), range(n_columns)):
        ax = axs[r][c]

        skill = fields[r][c]
        lon = skill.lon.values
        lat = skill.lat.values
        p = skill[variable].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=cmap,
            add_colorbar=False,
        )
        p.axes.coastlines(color=[0.2, 0.2, 0.2], linewidth=0.75)
        if f"{variable}_signif" in skill:
            ax.contourf(
                lon,
                lat,
                1 * skill[f"{variable}_signif"],
                [0, 0.5, 1],
                colors="none",
                hatches=[None, "///", None],
                transform=ccrs.PlateCarree(),
                extend="lower",
            )

        if headings is not None:
            ax.set_title(headings[r][c])

        title = ax.get_title()
        ax.set_title(f"{title} | {_get_verif_period(skill)}")

    fig.tight_layout()
    fig.patch.set_facecolor("w")

    if add_colorbar:
        # Colorbar with fixed physical height
        h = [Size.Fixed(figsize[0] / 12), Size.Fixed(figsize[0] - figsize[0] / 6)]
        v = [Size.Fixed(0), Size.Fixed(0.15)]
        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        #     fig.subplots_adjust(bottom=0.1)
        cbar_ax = fig.add_axes(
            divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
        )
        fig.colorbar(p, cax=cbar_ax, orientation="horizontal")

    return fig


def metrics(
    metrics,
    variable,
    headings=None,
    one_legend=True,
    shade_background=True,
    figsize=(15, 15),
):
    """
    Plot panels of skill scores. When multiple metrics are provided in a panel,
    cross are shown along the x-axis where both metrics are positive.

    Parameters
    ----------
    metrics : list
        List of size n_rows x n_columns containing dictionaries of the model
        metrics to plot in each panel, e.g.
        [[{"model1": {"rXY": metric1, "ri": metric2}, "model2": {"rXY": metric1, "ri": metric2}}]]
    variable : str
        The name of the variable
    headings : list
        List of the same size as fields containing the headins for each panel
    one_legend : boolean, optional
        If True, only add a legend to the first panel
    shade_background : boolean, optional
        Of True, shade the background either side of zero
    figsize : iterable of length 2
        The total size of the figure
    """

    fig = plt.figure(figsize=figsize)
    n_rows = len(metrics)
    n_columns = len(metrics[0])
    axs = fig.subplots(
        n_rows,
        n_columns,
        sharex=True,
        sharey=True,
    )
    if n_rows == 1:
        axs = [axs]
        if n_columns == 1:
            axs = [axs]
    elif n_columns == 1:
        axs = [[ax] for ax in axs]

    for r, c in itertools.product(range(n_rows), range(n_columns)):
        ax = axs[r][c]

        metric_dict = metrics[r][c]
        colors = ["C0", "C1", "C2", "C3", "C4"]
        colorcycler = cycle(colors)
        # spaces = [0,0.1,-0.1,0.2,-0.2]
        # spacecycler = cycle(spaces)
        plot_lines = []
        for model, model_metrics in metric_dict.items():
            color = next(colorcycler)
            lines = ["-", "--", "-.", ":"]
            linecycler = cycle(lines)
            metric_lines = []
            for metric_name, model_metric in model_metrics.items():
                line = next(linecycler)
                (p,) = model_metric[variable].plot(
                    ax=ax,
                    linestyle=line,
                    color=color,
                )
                model_metric[variable].where(model_metric[f"{variable}_signif"]).plot(
                    ax=ax, linestyle="none", marker="o", color=color
                )
                metric_lines.append(p)
            plot_lines.append(metric_lines)

            # if len(model_metrics) > 1:
            #     def logical_and(ds_1, ds_2):
            #         out = ds_1 & ds_2
            #         return out.where(out > 0)

            #     def reduction(ds):
            #         """How to select where crosses go when multiple metrics are provided"""
            #         return xr.where(
            #             (ds[variable] > 0) & ds[f"{variable}_signif"], True, False
            #         ).to_dataset(name=variable)
            #         # return ds[[variable]] > 0

            #     crosses = functools.reduce(
            #     logical_and, [reduction(ds) for ds in model_metrics.values()]
            #     ).dropna(dim="lead")
            #     space = next(spacecycler)
            #     ax.plot(crosses.lead,[space]*len(crosses.lead), marker='x', linestyle="none", color=color)

        if headings is not None:
            ax.set_title(headings[r][c])

        if ((one_legend is True) & (r == 0) & (c == 0)) | (one_legend is False):
            legend1 = ax.legend([l[0] for l in plot_lines], metric_dict.keys(), loc=1)
            has_multiple_lines = [len(l) > 1 for l in plot_lines]
            if any(has_multiple_lines):
                line_for_legend = [i for i, x in enumerate(has_multiple_lines) if x][0]
                leg = ax.legend(
                    plot_lines[line_for_legend],
                    metric_dict[list(metric_dict.keys())[0]].keys(),
                    loc=4,
                )
                for handle in leg.legendHandles:
                    handle.set_color("grey")
                ax.add_artist(legend1)

        if c == 0:
            if any(has_multiple_lines):
                ax.set_ylabel("Skill")
            else:
                # Assume all metrics are the same
                ax.set_ylabel(list(model_metrics.keys())[0])
        else:
            ax.set_ylabel("")

        if r == (n_rows - 1):
            ax.set_xlabel("Final lead month of forecast period")
        else:
            ax.set_xlabel("")

        ax.grid(True)
        ax.set_ylim(-1, 1)

        if shade_background:
            neg_color = "m"
            pos_color = "g"
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.fill_between(
                xlim, [ylim[0], ylim[0]], color=neg_color, alpha=0.1, zorder=-1
            )
            ax.fill_between(
                xlim, [ylim[1], ylim[1]], color=pos_color, alpha=0.1, zorder=-1
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    plt.tight_layout()
    fig.patch.set_facecolor("w")
    return fig
