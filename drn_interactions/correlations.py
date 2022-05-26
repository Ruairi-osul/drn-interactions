"""Transforms for correlating neurons
"""

import numpy as np
import pandas as pd
from typing import Optional
from itertools import combinations_with_replacement
from drn_interactions.transforms import (
    bin_spikes_interval,
    gaussian_smooth,
    pivot,
)
from neurobox.compose import Pipeline
from spiketimes.df.surrogates import shuffled_isi_spiketrains_by
from spiketimes.utils import p_adjust
from tqdm import tqdm
from neurobox.correlations import pairwise_correlation, correlation_matrix_to_tidy


def pairwise_correlation_spikes(
    df_spikes: pd.DataFrame,
    bin_width: float,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
    t_start: float = 0,
    rectify: bool = False,
    zero_diag: bool = True,
    fillna: Optional[float] = None,
    sigma: float = 0,
) -> pd.DataFrame:
    pipe = Pipeline(
        [
            (
                bin_spikes_interval,
                dict(
                    bin_width=bin_width,
                    spiketimes_col=spiketimes_col,
                    neuron_col=neuron_col,
                    t_start=t_start,
                ),
            ),
            (
                pivot,
                dict(neuron_col=neuron_col, value_col="spike_count", time_col="time"),
            ),
            (gaussian_smooth, dict(sigma=sigma)),
            (
                pairwise_correlation,
                dict(rectify=rectify, zero_diag=zero_diag, fillna=fillna),
            ),
        ]
    )
    return pipe.transform(df_spikes)


def shuffled_isi_correlation_test(
    df_spikes: pd.DataFrame,
    bin_width: float,
    n_boot: int,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
    t_start: float = 0,
    fillna: Optional[float] = 0,
    sigma: float = 0,
    show_progress: bool = False,
    rectify: bool = False,
) -> pd.DataFrame:

    corr_getter = Pipeline(
        [
            (
                pairwise_correlation_spikes,
                dict(
                    bin_width=bin_width,
                    neuron_col=neuron_col,
                    spiketimes_col=spiketimes_col,
                    t_start=t_start,
                    sigma=sigma,
                    rectify=rectify,
                    zero_diag=True,
                    fillna=fillna,
                ),
            ),
            (correlation_matrix_to_tidy, dict()),
        ]
    )
    obs = corr_getter.transform(df_spikes)

    reps = np.empty((obs.shape[0], n_boot))
    p = np.empty(obs.shape[0])
    if show_progress:
        range_iter = tqdm(range(n_boot))
    else:
        range_iter = range(n_boot)
    for i in range_iter:
        df_shuffled = shuffled_isi_spiketrains_by(df_spikes, by_col="neuron_id")
        res = corr_getter.transform(df_shuffled).rename(columns={"value": "bs_rep"})
        # order as in observed
        res = pd.merge(
            obs[["neuron_combination", "neuron_1", "neuron_2"]], res, how="left"
        )
        reps[:, i] = res["bs_rep"].values

    # calculate p values
    obs_values = obs["value"].values
    for i in range(obs.shape[0]):
        p[i] = np.nanmean(np.abs(reps[i]) > np.abs(obs_values[i])) * 2
    return obs.assign(p=p, p_adj=p_adjust(p))


def get_combs(
    df_combs,
    neurons,
    neurons_col="neuron_id",
    neuron_type_col="cluster",
    combs_1="neuron_1",
    combs_2="neuron_2",
):
    def which_comb(row, combs, combs_1, combs_2):
        n1 = row[combs_1]
        n2 = row[combs_2]
        for comb in combs:
            if set([n1, n2]) == set(comb):
                return comb
        return np.nan

    df_combs = (
        df_combs.merge(neurons, left_on=combs_1, right_on=neurons_col)
        .drop(neurons_col, axis=1)
        .rename(columns={neuron_type_col: f"n1_{neuron_type_col}"})
    )
    df_combs = (
        df_combs.merge(neurons, left_on=combs_2, right_on=neurons_col)
        .drop(neurons_col, axis=1)
        .rename(columns={neuron_type_col: f"n2_{neuron_type_col}"})
    )

    neuron_types = neurons["cluster"].unique()
    combs = list(combinations_with_replacement(neuron_types, r=2))
    df_combs["comb"] = df_combs.apply(
        which_comb,
        combs=combs,
        combs_1=f"n1_{neuron_type_col}",
        combs_2=f"n2_{neuron_type_col}",
        axis=1,
    )
    df_combs["comb"] = df_combs["comb"].apply(lambda x: "-".join(x))
    return df_combs
