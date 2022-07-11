from typing import Optional, Tuple
from drn_interactions.transforms import SpikesHandler
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def population_raster(
    session_name: str,
    block: str = "chal",
    clusters: Optional[pd.DataFrame] = None,
    bin_width: float = 1,
    figsize: Tuple[float, float] = (5, 3),
    title: bool = True,
    t_start: float = -600,
    t_stop: float = 1200,
    tfidf: bool = False,
) -> plt.Axes:
    _, ax = plt.subplots(figsize=figsize)
    df = SpikesHandler(
        block=block,
        session_names=[session_name],
        bin_width=bin_width,
        t_start=t_start,
        t_stop=t_stop,
    ).binned_piv
    if clusters is not None:
        session_neurons = df.columns
        clusters = clusters.loc[lambda x: x.neuron_id.isin(session_neurons)]
        clusters = clusters.sort_values("wf_3")
        idx = clusters.neuron_id.values.tolist()
        df = df[idx]
    vals = df.values.T
    if tfidf:
        vals = TfidfTransformer().fit_transform(vals.T).toarray().T
    sns.heatmap(
        data=vals,
        cmap="Greys",
        ax=ax,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        robust=True,
    )
    ax.axis("off")
    if title:
        ax.set_title(session_name)
    return ax
