from citalopram_project.correlations import pairwise_correlation
from citalopram_project.ensemble.humphries import (
    humphries_ensemble,
    communities_test,
    modularity_test,
)
import numpy as np
import pandas as pd
from typing import Optional


def find_ensembles(
    df,
    time_col="trial_num",
    value_col="counts",
    cell_col="neuron_id",
    n_boot_mod=20,
    n_boot_coms=1000,
):
    df_corr = pairwise_correlation(
        df=df,
        cell_col=cell_col,
        value_col=value_col,
        time_col=time_col,
        rectify=True,
        fillna=0,
        return_tidy=False,
    )
    mod, coms, _ = humphries_ensemble(df_corr, n_init=30)
    if np.isnan(mod):
        modularity_pval = np.nan
        df_ensemble_stats = pd.DataFrame(
            {
                "modularity": np.nan,
                "modularity_pval": np.nan,
                "ensemble": np.nan,
                "size": np.nan,
                "score": np.nan,
                "score_pval": np.nan,
                "simmilarity": np.nan,
            },
            index=[0],
        )
        df_ensembles = pd.DataFrame()
        return df_ensembles, df_ensemble_stats
    _, _, null_modularity_distrobution = modularity_test(
        df,
        cell_col=cell_col,
        value_col=value_col,
        time_col=time_col,
        fillna=0,
        n_boot=n_boot_mod,
    )
    modularity_pval = np.nanmean(null_modularity_distrobution > mod)
    com_scores, com_pvals, com_sims = communities_test(
        df_corr, coms, n_boot=n_boot_coms
    )
    df_ensembles = _create_ensemble_df(coms)
    df_ensemble_stats = _create_ensemble_stats(
        modularity=mod,
        modularity_pval=modularity_pval,
        communities=coms,
        community_scores=com_scores,
        community_pvals=com_pvals,
        community_similarities=com_sims,
    )
    return df_ensembles, df_ensemble_stats


def _create_ensemble_stats(
    mod,
    communities,
    community_scores,
    community_pvals,
    community_similarities,
):
    sers = []
    for i, (stats) in enumerate(
        zip(communities, community_scores, community_pvals, community_similarities)
    ):
        com, score, pval, sim = stats
        ser = pd.Series(
            {
                "modularity": mod,
                "ensemble": i,
                "size": len(com),
                "score": score,
                "score_pval": pval,
                "simmilarity": sim,
            }
        )
        sers.append(ser)
    return pd.DataFrame(sers)


def _create_ensemble_df(coms):
    frames = []
    for i, com in enumerate(coms):
        df = pd.DataFrame({"ensemble": i, "neuron_id": com})
        frames.append(df)
    return pd.concat(frames)


def get_ensemble_id(ensemble_stats_df, ensemble_df):
    ensemble_stats_df = (
        ensemble_stats_df.reset_index()
        .drop("index", axis=1)
        .reset_index()
        .rename(columns={"index": "ensemble_id"})
    )
    ensemble_df = pd.merge(
        ensemble_stats_df[["session_name", "ensemble", "ensemble_id"]], ensemble_df
    )
    ensemble_stats_df = ensemble_stats_df.drop("ensemble", axis=1)
    ensemble_df = ensemble_df.drop("ensemble", axis=1)
    return ensemble_stats_df, ensemble_df


def get_ensemble_sig(ensemble_stats_df, min_similarity=0.1, min_size=3):
    ensemble_stats_df = ensemble_stats_df.assign(
        ensemble_sig=np.where(
            (ensemble_stats_df["score_pval"] < 0.05)
            & (ensemble_stats_df["simmilarity"] > min_similarity)
            & (~np.isnan(ensemble_stats_df["modularity"]))
            & (ensemble_stats_df["size"] >= min_size),
            True,
            False,
        )
    )
    return ensemble_stats_df


def drop_non_sig_ensembles(ensemble_stats_df, ensemble_df):
    sig_ensembles = ensemble_stats_df.loc[
        lambda x: x.ensemble_sig == True
    ].ensemble_id.unique()
    dropped = ensemble_df.loc[lambda x: x.ensemble_id.isin(sig_ensembles)]
    return dropped
