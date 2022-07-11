from typing import Optional, Dict, Callable, Iterable, Sequence
from umap import UMAP
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def umap_spikes(
    df_piv: pd.DataFrame,
    y: Optional[pd.Series] = None,
    umap_kwargs: Optional[Dict] = None,
    y_transformer: Optional[Callable] = None,
):
    if umap_kwargs is None:
        umap_kwargs = {}
    y_vals = y
    if y_transformer is not None:
        y_vals = y_transformer(y)
    mod = UMAP(**umap_kwargs)
    mod = mod.fit(df_piv.values, y=y_vals)
    df_fit = pd.DataFrame(mod.embedding_, columns=["d1", "d2"], index=df_piv.index)
    if y is not None:
        df_fit = df_fit.join(y)
    df_fit = df_fit.reset_index()
    return df_fit
