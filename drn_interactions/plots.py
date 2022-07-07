import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


PAL_GREY_BLACK = sns.color_palette(["black", "silver"])


def circular_hist(
    x,
    bins=16,
    ax=None,
    density=True,
    offset=0,
    gaps=True,
    linewidth=1,
    edgecolor="black",
    fill=False,
    **kwargs
):
    """
    Produce a circular histogram of angles on ax.

    From stackoverflow: https://stackoverflow.com/a/55067613/7993732

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** 0.5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(
        bins[:-1],
        radius,
        zorder=1,
        align="edge",
        width=widths,
        edgecolor=edgecolor,
        linewidth=linewidth,
        fill=fill,
        **kwargs,
    )

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    ax.set_xticklabels(
        ["0", "", "0.5 $\cdot\pi$", "", "$\pi$", "", "-0.5 $\cdot\pi$", "", ""]
    )
    return ax


from matplotlib.ticker import (
    Locator,
    Formatter,
    NullLocator,
    NullFormatter,
    AutoLocator,
)
from typing import Optional, Sequence, Tuple, Dict
from matplotlib.colorbar import Colorbar
import pandas as pd
from scipy.stats import zscore


def heatmap(
    df_binned_piv: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    locater_x: Locator = NullLocator(),
    formater_x: Formatter = NullFormatter(),
    locater_y: Locator = AutoLocator(),
    formater_y: Optional[Formatter] = None,
    yticklabels_kwargs: Optional[Dict] = None,
    heatmap_kwargs: Optional[Dict] = None,
    tick_params: Optional[Dict] = None,
    cbar_tick_locater: Locator = AutoLocator(),
    cbar_ticklabel_kwargs: Optional[Dict] = None,
    cbar_title_kwargs: Optional[Dict] = None,
    cbar_tick_params: Optional[Dict] = None,
) -> Tuple[plt.Axes, Optional[Colorbar]]:
    heatmap_kwargs = {} if heatmap_kwargs is None else heatmap_kwargs
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    X = df_binned_piv.values
    sns.heatmap(X, ax=ax, **heatmap_kwargs)
    ax.xaxis.set_major_locator(locater_x)
    ax.xaxis.set_major_formatter(formater_x)
    ax.yaxis.set_major_locator(locater_y)
    if formater_y is not None:
        ax.yaxis.set_major_formatter(formater_y)
    if yticklabels_kwargs:
        ax.set_yticklabels(**yticklabels_kwargs)
    if tick_params:
        ax.tick_params(**tick_params)

    try:
        cbar: Colorbar = ax.collections[0].colorbar
        if cbar_title_kwargs:
            cbar.ax.set_title(**cbar_title_kwargs)
        cbar.set_ticks(cbar_tick_locater)
        if cbar_ticklabel_kwargs:
            cbar.ax.set_yticklabels(**cbar_ticklabel_kwargs)
        if cbar_tick_params:
            cbar.ax.tick_params(**cbar_tick_params)
        return ax, cbar
    except AttributeError:
        return ax, None


def similarity_map(
    df_binned_piv: pd.DataFrame,
    z: bool = False,
    ax: Optional[plt.Axes] = None,
    locater_x: Locator = NullLocator(),
    formater_x: Formatter = NullFormatter(),
    locater_y: Locator = AutoLocator(),
    formater_y: Optional[Formatter] = None,
    yticklabels_kwargs: Optional[Dict] = None,
    heatmap_kwargs: Optional[Dict] = None,
    tick_params: Optional[Dict] = None,
    cbar_tick_locater: Locator = AutoLocator(),
    cbar_ticklabel_kwargs: Optional[Dict] = None,
    cbar_title_kwargs: Optional[Dict] = None,
    cbar_tick_params: Optional[Dict] = None,
) -> Tuple[plt.Axes, Optional[Colorbar]]:
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    X = df_binned_piv.values if not z else df_binned_piv.apply(zscore)
    sim = 1 - pairwise_distances(X, metric="cosine")
    df_sim = pd.DataFrame(sim, index=df_binned_piv.index, columns=df_binned_piv.index)

    ax, cbar = heatmap(
        df_binned_piv=df_sim,
        ax=ax,
        locater_x=locater_x,
        formater_x=formater_x,
        locater_y=locater_y,
        formater_y=formater_y,
        yticklabels_kwargs=yticklabels_kwargs,
        heatmap_kwargs=heatmap_kwargs,
        tick_params=tick_params,
        cbar_tick_locater=cbar_tick_locater,
        cbar_ticklabel_kwargs=cbar_ticklabel_kwargs,
        cbar_title_kwargs=cbar_title_kwargs,
        cbar_tick_params=cbar_tick_params,
    )
    return ax, cbar


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer


def long_raster(
    df_binned_piv: pd.DataFrame,
    tfidf: bool = False,
    minmax: bool = False,
    ax: Optional[plt.Axes] = None,
    locater_x: Locator = NullLocator(),
    formater_x: Formatter = NullFormatter(),
    locater_y: Locator = AutoLocator(),
    formater_y: Optional[Formatter] = None,
    yticklabels_kwargs: Optional[Dict] = None,
    heatmap_kwargs: Optional[Dict] = None,
    tick_params: Optional[Dict] = None,
    cbar_tick_locater: Locator = AutoLocator(),
    cbar_ticklabel_kwargs: Optional[Dict] = None,
    cbar_title_kwargs: Optional[Dict] = None,
    cbar_tick_params: Optional[Dict] = None,
) -> Tuple[plt.Axes, Optional[Colorbar]]:
    if tfidf:
        X = TfidfTransformer().fit_transform(df_binned_piv.values).toarray()
    elif minmax:
        X = MinMaxScaler().fit_transform(df_binned_piv.values)
    else:
        X = df_binned_piv.values

    df_spikes = pd.DataFrame(
        X, index=df_binned_piv.index, columns=df_binned_piv.columns
    ).transpose()
    ax, cbar = heatmap(
        df_binned_piv=df_spikes,
        ax=ax,
        locater_x=locater_x,
        formater_x=formater_x,
        locater_y=locater_y,
        formater_y=formater_y,
        yticklabels_kwargs=yticklabels_kwargs,
        heatmap_kwargs=heatmap_kwargs,
        tick_params=tick_params,
        cbar_tick_locater=cbar_tick_locater,
        cbar_ticklabel_kwargs=cbar_ticklabel_kwargs,
        cbar_title_kwargs=cbar_title_kwargs,
        cbar_tick_params=cbar_tick_params,
    )
    return ax, cbar


def state_indicator(
    states: pd.Series,
    order: Sequence,
    ax: Optional[plt.Axes] = None,
    margins: float = 0,
    despine: bool = True,
    locater_x: Locator = AutoLocator(),
    formater_x: Optional[Formatter] = None,
    tick_params: Optional[Dict] = None,
    plot_kwargs: Optional[Dict] = None,
    ytickslabs: Optional[Sequence] = None,
):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ytickslabs = order if ytickslabs is None else ytickslabs
    mapper = {val: i for i, val in enumerate(order)}
    vals = states.map(mapper)
    ax.plot(states.index, vals, color="black", **plot_kwargs)
    ax.xaxis.set_major_locator(locater_x)
    if formater_x is not None:
        ax.xaxis.set_major_formatter(formater_x)
    ax.set_yticks(range(len(order)))

    ax.set_yticklabels(ytickslabs, rotation=0)
    if tick_params is not None:
        ax.tick_params(**tick_params)
    ax.margins(x=margins)
    if despine:
        sns.despine(ax=ax, left=True)
    return ax


def plot_umap(
    df_umap: pd.DataFrame,
    y: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    dim_cols: Sequence[str] = ("d1", "d2"),
    edgecolor="black",
    alpha=0.2,
    remove_legend: bool = True,
    despine: bool = True,
    formater_x: Optional[Formatter] = None,
    formater_y: Optional[Formatter] = None,
    locater_x: Locator = AutoLocator(),
    locater_y: Locator = AutoLocator(),
    tick_params: Optional[Dict] = None,
    scatter_kwargs: Optional[Dict] = None,
):
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    sns.scatterplot(
        x=dim_cols[0],
        y=dim_cols[1],
        hue=y,
        data=df_umap,
        ax=ax,
        edgecolor=edgecolor,
        alpha=alpha,
        **scatter_kwargs,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.xaxis.set_major_locator(locater_x)
    if formater_x is not None:
        ax.xaxis.set_major_formatter(formater_x)
    ax.yaxis.set_major_locator(locater_y)
    if formater_y is not None:
        ax.yaxis.set_major_formatter(formater_y)
    if tick_params is not None:
        ax.tick_params(**tick_params)

    if remove_legend:
        ax.legend().remove()
    if despine:
        sns.despine(ax=ax)
    return ax
