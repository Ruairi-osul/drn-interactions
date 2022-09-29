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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances


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
    if locater_x is not NullLocator:
        ax.xaxis.set_major_locator(locater_x)
    if formater_x is not None:
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
    plot_kwargs["color"] = plot_kwargs.get("color", "k")
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ytickslabs = order if ytickslabs is None else ytickslabs
    mapper = {val: i for i, val in enumerate(order)}
    vals = states.map(mapper)
    ax.plot(states.index, vals, **plot_kwargs)
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
