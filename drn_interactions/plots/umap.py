from matplotlib.ticker import Locator, Formatter, AutoLocator
from typing import Optional, Sequence, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
