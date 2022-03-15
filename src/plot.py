import itertools

import cftime

import numpy as np
import xarray as xr

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import cartopy
import cartopy.crs as ccrs

cartopy.config["pre_existing_data_dir"] = "../../data/cartopy-data"
cartopy.config["data_dir"] = "../../data/cartopy-data"


def hindcasts(hcsts, obsvs=None, hists=None, shade=False):
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

    from itertools import cycle
    from matplotlib.pyplot import cm
    from matplotlib.dates import date2num

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

    fig = plt.figure(figsize=(10, n_vars * 4))
    axs = fig.subplots(n_vars, 1, sharex=True)
    if n_vars == 1:
        axs = [axs]

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
                axs[a].plot(
                    h[hcst_time][0], h[0], color=c, marker="o", label="_nolabel_"
                )
                axs[a].plot(h[hcst_time], h, color=c, linestyle="-", label=label)
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
        linecycler = cycle(lines)
        for name, hist in hists.items():
            line = next(linecycler)
            for a, var in enumerate(hist.data_vars):
                axs[a].plot(
                    hist.time, hist[var], color="grey", label=name, linestyle=line
                )

    # Format plots
    ticks = xr.cftime_range(start=xlim[0], end=xlim[-1], freq="2AS", calendar="julian")
    xlim = (ticks.shift(-1, "AS")[0], ticks.shift(2, "AS")[-1])
    for a, var in enumerate(hcst.data_vars):
        axs[a].set_xticks(ticks.values)
        axs[a].set_xticklabels(ticks.year)
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
    fig.patch.set_facecolor("w")

    return fig


def skill_maps(
    fields, variable, vrange, headings=None, group_rows_by=1, figsize=(15, 9)
):
    """
    Plot panels of skill score maps

    Parameters
    ----------
    fields : list
        List of size n_rows x n_coloumns containing the fields to plot
    variable : str
        The name of the variable
    vrange : iterable of length 2
        The vmin and vmax values for all panels
    figsize : iterable of length 2
        The total size of the figure
    """
    
    def _get_verif_period(ds):
        """Return a string of the verification period for a skill metric"""

        return (
            f"{ds.attrs['verification period start'][:4]}-"
            f"{ds.attrs['verification period end'][:4]}"
        )

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

    cmap = cm.get_cmap("RdBu_r", 10)

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
        ax.contourf(
            lon,
            lat,
            1 * skill[f"sst_signif"],
            [0, 0.5, 1],
            colors="none",
            hatches=[None, "///", None],
            transform=ccrs.PlateCarree(),
        )
        
        if headings is not None:
            ax.set_title(headings[r][c])
            
        title = ax.get_title()
        ax.set_title(f"{title} | {_get_verif_period(skill)}")

    fig.tight_layout()
    return fig
