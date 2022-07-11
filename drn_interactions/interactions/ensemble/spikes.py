import pandas as pd


def pairwise_correlation_spikes(
    df: pd.DataFrame,
    spiketrain_col: str = "spiketimes",
    spiketimes_col: str = "neuron_id",
    bin_width: float = 1,
    fillna: float = 0,
    t_start: float = 0,
    rectify: bool = True,
    gaussian_sigma: Optional[float] = None,
    return_tidy: bool = True,
) -> pd.DataFrame:
    # TODO
    fs = 1 / bin_width
    df_binned = ifr_by(
        df,
        spiketrain_col=spiketrain_col,
        spiketimes_col=spiketimes_col,
        fs=fs,
        sigma=gaussian_sigma,
        t_start=t_start,
    )
    df_corr = pairwise_correlation(
        df_binned,
        cell_col=spiketrain_col,
        time_col="time",
        value_col="ifr",
        return_tidy=return_tidy,
        rectify=rectify,
        fillna=fillna,
    )
    return df_corr


def humphries_ensemble_spikes(
    df_spikes,
    n_init=30,
    bin_width=1,
    cell_col="neuron_id",
    spiketimes_col="spiketimes",
    fillna=0,
    t_start=0,
):
    # TODO
    fs = 1 / bin_width
    df_binned = ifr_by(
        df_spikes,
        spiketrain_col=cell_col,
        spiketimes_col=spiketimes_col,
        fs=fs,
        sigma=0.5,
        t_start=t_start,
    )
    df_corr = pairwise_correlation(
        df_binned,
        cell_col=cell_col,
        time_col="time",
        value_col="ifr",
        return_tidy=False,
        rectify=True,
        fillna=fillna,
    )
    modularity, communities, pred = humphries_ensemble(df_corr, n_init=n_init)
    return modularity, communities, pred


def communities_test_spikes(
    df_spikes,
    communities,
    bin_width=1,
    cell_col="neuron_id",
    spiketimes_col="spiketimes",
    t_start=0,
    fillna=0,
    n_boot=1000,
):
    fs = 1 / bin_width
    df_binned = ifr_by(
        df_spikes,
        spiketrain_col=cell_col,
        spiketimes_col=spiketimes_col,
        fs=fs,
        sigma=0.5,
        t_start=t_start,
    )
    df_corr = pairwise_correlation(
        df_binned,
        cell_col=cell_col,
        time_col="time",
        value_col="ifr",
        return_tidy=False,
        rectify=True,
        fillna=fillna,
    )
    community_scores, community_pvals, community_similarities = communities_test(
        df_corr, communities, n_boot=n_boot
    )
    return community_scores, community_pvals, community_similarities


def modularity_test_spikes(
    df_spikes,
    bin_width=1,
    t_start=0,
    cell_col="neuron_id",
    spiketimes_col="spiketimes",
    n_boot=20,
    fillna=0,
):
    """Perform a bootstrap test of modularity using humphries community detection algorithm on spiketimes data

    Args:
        df_spikes (pd.DataFrame): pandas dataframe containing spiketimes by cell
        bin_width (float, optional): Size of the interval used to bin spikes. Defaults to 1.
        t_start (float, optional): Time in seconds of the start point of the . Defaults to 0.
        cell_col (str, optional): Column in df_spikes containing cell labels. Defaults to "neuron_id".
        spiketimes_col (str, optional): Column in df_spikes containing spike times. Defaults to "spiketimes".
        n_boot (int, optional): Number of bootstrap replicates to draw. Defaults to 20.
        fillna (int, optional): If set to zero, will fill bins with missing spikes as zeros. Defaults to 0.
    """

    def _null_modularity_distrobution_spikes(
        df_spikes,
        cell_col="neuron_id",
        spiketimes_col="spiketimes",
        bin_width=1,
        t_start=0,
        n_boot=20,
        fillna=0,
    ):
        fs = 1 / bin_width
        mods = []
        for _ in range(n_boot):
            df_surr = shuffled_isi_spiketrains_by(
                df_spikes, by_col=cell_col, spiketimes_col=spiketimes_col, n=1
            )
            df_binned = ifr_by(
                df_surr, spiketrain_col=cell_col, fs=fs, sigma=0.5, t_start=t_start
            )
            c_sur = pairwise_correlation(
                df_binned,
                value_col="ifr",
                cell_col=cell_col,
                return_tidy=False,
                rectify=True,
                fillna=fillna,
            )
            modularity1_sur, _, _ = humphries_ensemble(c_sur)
            mods.append(modularity1_sur)
        mods = np.array(mods)
        return mods

    fs = 1 / bin_width
    df_binned = ifr_by(
        df_spikes,
        spiketrain_col=cell_col,
        fs=fs,
        sigma=0.5,
        t_start=t_start,
        spiketimes_col=spiketimes_col,
    )
    df_corr = pairwise_correlation(
        df_binned,
        cell_col=cell_col,
        time_col="time",
        value_col="ifr",
        return_tidy=False,
        rectify=True,
        fillna=fillna,
    )
    empirical_mod, _, _ = humphries_ensemble(df_corr)
    if np.isnan(empirical_mod):
        return np.nan, np.nan, np.array([])
    null_distrobution = _null_modularity_distrobution_spikes(
        df_spikes,
        cell_col=cell_col,
        spiketimes_col=spiketimes_col,
        bin_width=bin_width,
        n_boot=n_boot,
        fillna=fillna,
    )
    p_value = np.nanmean(null_distrobution > empirical_mod)
    return empirical_mod, p_value, null_distrobution


def find_ensembles_spikes(
    df_spikes,
    bin_width=1,
    cell_col="neuron_id",
    spiketimes_col="spiketimes",
    fillna=0,
    t_start=0,
    n_boot_mod=20,
    n_boot_coms=1000,
):

    mod, coms, _ = humphries_ensemble_spikes(
        df_spikes=df_spikes,
        bin_width=bin_width,
        cell_col=cell_col,
        spiketimes_col=spiketimes_col,
        fillna=fillna,
        t_start=t_start,
    )
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
    _, _, null_dist = modularity_test_spikes(
        df_spikes=df_spikes,
        bin_width=bin_width,
        cell_col=cell_col,
        spiketimes_col=spiketimes_col,
        fillna=fillna,
        t_start=t_start,
        n_boot=n_boot_mod,
    )
    modularity_pval = np.nanmean(null_dist > mod)
    (
        community_scores,
        community_pvals,
        community_similarities,
    ) = communities_test_spikes(
        df_spikes,
        coms,
        bin_width=bin_width,
        cell_col=cell_col,
        spiketimes_col=spiketimes_col,
        t_start=t_start,
        fillna=fillna,
        n_boot=n_boot_coms,
    )
    df_ensembles = _create_ensemble_df(coms)
    df_ensemble_stats = _create_ensemble_stats(
        modularity=mod,
        modularity_pval=modularity_pval,
        communities=coms,
        community_scores=community_scores,
        community_pvals=community_pvals,
        community_similarities=community_similarities,
    )
    return df_ensembles, df_ensemble_stats

