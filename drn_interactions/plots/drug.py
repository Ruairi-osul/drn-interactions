from typing import Optional, Tuple
from drn_interactions.transforms import SpikesHandler
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Sequence, Any


def population_raster(
    session_name: str,
    block: str = "chal",
    clusters: Optional[pd.DataFrame] = None,
    bin_width: float = 1,
    ax: Optional[plt.Axes] = None,
    cbar_ax: Optional[plt.Axes] = None,
    idx: Optional[Sequence[Any]] = None,
    title: bool = True,
    t_start: float = -600,
    t_stop: float = 1200,
    tfidf: bool = False,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    if cbar_ax is None:
        plot_cbar = False
    else:
        plot_cbar = True
    df = SpikesHandler(
        block=block,
        session_names=[session_name],
        bin_width=bin_width,
        t_start=t_start,
        t_stop=t_stop,
    ).binned_piv

    if idx is not None:
        df = df.transpose().reindex(idx).dropna().transpose()
    vals = df.values.T
    if tfidf:
        vals = TfidfTransformer().fit_transform(vals.T).toarray().T
    sns.heatmap(
        data=vals,
        cmap="Greys",
        cbar_ax=cbar_ax,
        ax=ax,
        cbar=plot_cbar,
        xticklabels=False,
        yticklabels=False,
        robust=True,
    )
    ax.axis("off")
    if title:
        ax.set_title(session_name)
    return ax
