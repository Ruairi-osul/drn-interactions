"""Contains functions used in data processing pipelines

All functions take a pd.DataFrame as their first argument and return a pd.DataFrame
"""

from spiketimes.df.binning import binned_spiketrain_bins_provided, binned_spiketrain
from spiketimes.df.alignment import align_around_by
from spiketimes.df.statistics import mean_firing_rate_by
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, zmap
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd


def bin_spikes(
    df: pd.DataFrame,
    bin_width: float,
    t_before: float,
    t_max: float,
    spikes_col: str = "spiketimes",
) -> pd.DataFrame:
    """Bin spikes into bins created as specified

    Args:
        df (pd.DataFrame): DataFrame containing spikes
        bin_width (float): Binwidth
        t_before (float): Starting bin
        t_max (float): Ending bin (does not include this bin)
        spikes_col (str, optional): Column in df containing spiketimes. Defaults to "spiketimes".

    Returns:
        pd.DataFrame: A dataframe containing columns {'neuron_id', 'bin', 'counts'}
    """
    bins = np.round(np.arange(t_before, t_max, bin_width), 3)
    return binned_spiketrain_bins_provided(
        df, bins=bins, spiketimes_col=spikes_col, spiketrain_col="neuron_id"
    )


def bin_spikes_interval(
    df: pd.DataFrame,
    bin_width: float,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
    t_start: Optional[float] = None,
) -> pd.DataFrame:
    """Bins a df containing spikes at a regular binwidth

    Args:
        df (pd.DataFrame): Input DF
        bin_width (float): Length of interval used for binning equivilant to 1/sampling_rate
        spiketimes_col (str, optional): Name of column containing spiketimes. Defaults to "spiketimes".
        neuron_col (str, optional): Name of column containing neuron identifiers. Defaults to "neuron_id".
        t_start (Optional[float], optional): If specified, marks the first bin. Otherwise is the first spike for each neuron. Defaults to None.

    Returns:
        pd.DataFrame: Binned DF with columns {'neuron_id', 'time', 'spike_count'}
    """
    return binned_spiketrain(
        df,
        spiketimes_col=spiketimes_col,
        spiketrain_col=neuron_col,
        fs=1 / bin_width,
        t_start=t_start,
    )


def align_spikes_to_events(
    df_data: pd.DataFrame,
    df_events: pd.DataFrame,
    t_before: Optional[float] = None,
    session_col: str = "session_name",
) -> pd.DataFrame:
    """Aligns all spikes in df_data to events in df_events by session_col

    Args:
        df_data (pd.DataFrame): DataFrame containing spikes
        df_events (pd.DataFrame): DataFrame containing events (with label 'event_s')
        t_before (Optional[float], optional): If specified, spikes occuing in a window of 't_before' seconds before events will be negatively aligned to them. Defaults to None - Zero.
        session_col (str): Name of column in DataFrames containing session labels. Defaults to 'session_name'.

    Returns:
        pd.DataFrame: DataFrame whose spikes (in 'algined' col) are aligned to events by session
    """
    df = align_around_by(
        df_data=df_data,
        df_data_data_colname="spiketimes",
        df_data_group_colname=session_col,
        df_events=df_events,
        df_events_event_colname="event_s",
        df_events_group_colname=session_col,
        t_before=t_before,
    )
    return df


def pivot(
    df: pd.DataFrame,
    neuron_col: str = "neuron_id",
    value_col: str = "value",
    time_col: str = "time",
) -> pd.DataFrame:
    """Pivot a DataFrame to have one neuron per column and one timepoint per row

    Args:
        df (pd.DataFrame): DataFrame
        neuron_col (str, optional): Column containing neuron identifiers. Defaults to "neuron_id".
        value_col (str, optional): Column containing values. Defaults to "value".
        time_col (str, optional): Column containing time identifiers. Defaults to "time".

    Returns:
        pd.DataFrame: DataFrame with one column per neuron and one row per timepoint
    """
    return pd.pivot(df, values=value_col, index=time_col, columns=neuron_col)


def rename(df: pd.DataFrame, **column_mapping: Dict[str, str]):
    """Rename a column of the input DataFrame

    Args:
        df (pd.DataFrame): Input DF

    Returns:
        pd.DataFrame: DataFrame with updated columns
    """
    return df.rename(columns=column_mapping)


def drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop columns from input DataFrame

    Args:
        df (pd.DataFrame): DF
        cols (List[str]): Columns to drop

    Returns:
        pd.DataFrame: Input DataFrame with provided columns dropped
    """
    return df.drop(cols, axis=1)


def dropna_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NA values

    Args:
        df (pd.DataFrame): Input DF

    Returns:
        pd.DataFrame: DF without NA rows
    """
    return df.dropna()


def exlude_using_spike_rate_long(
    df: pd.DataFrame,
    min_rate: float,
    baseline_before: Optional[float] = None,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
) -> pd.DataFrame:
    """Exclude neurons whose spike rate is below a minimum threshold. Where input df is in long format

    Args:
        df (pd.DataFrame): Input DF in long format (i.e. one row per time point per neuron)
        min_rate (float): Threshold cutoff
        baseline_before (Optional[float], optional): If specified, calculates spike rate using data occuring before this point. Defaults to None.
        spiketimes_col (str, optional): Column containing spiketimes. Defaults to "spiketimes".
        neuron_col (str, optional): Column containing neuron identifiers. Defaults to "neuron_id".

    Returns:
        pd.DataFrame: DF with low spike rate neurons excluded
    """
    if baseline_before is not None:
        max_spike = baseline_before
    else:
        max_spike = df[spiketimes_col].max()
    idx = (
        mean_firing_rate_by(
            df.loc[lambda x: x[spiketimes_col] <= max_spike],
            spiketrain_col=neuron_col,
            spiketimes_col=spiketimes_col,
        )
        .loc[lambda x: x.mean_firing_rate > min_rate][neuron_col]
        .values
    )
    return df.loc[lambda x: x[neuron_col].isin(idx)]


def exclude_baseline(
    df: pd.DataFrame, time_col: str, baseline_before: float = 0,
) -> pd.DataFrame:
    """Exclude data from baseline

    Args:
        df (pd.DataFrame): Input DF
        time_col (str): Column containing time values
        baseline_before (float, optional): Timepoint at which baseline ends. Defaults to 0.

    Returns:
        pd.DataFrame: DF with baseline excluded
    """
    return df.loc[lambda x: x[time_col] >= baseline_before]


def standardize(
    df: pd.DataFrame, baseline_before: Optional[float] = None
) -> pd.DataFrame:
    """Standardize each column in a dataframe

    Args:
        df (pd.DataFrame): DF in wide format [time, neurons]
        baseline_before (Optional[float], optional): If specified, calculated zscores on values occuring before this point. Defaults to None.

    Returns:
        pd.DataFrame: Standarsized DF
    """

    def _standardize_col(col, baseline_before=None):
        if baseline_before is not None:
            return zmap(col, col.loc[lambda x: x.index < baseline_before])
        else:
            return zscore(col)

    return df.apply(_standardize_col, baseline_before=baseline_before)


def get_group_df(
    df: pd.DataFrame,
    df_neurons: pd.DataFrame,
    group: str,
    group_col: str = "group",
    neuron_col: str = "neuron_id",
) -> pd.DataFrame:
    """Subset columns of a DataFrame to only contain neurons of a specified group

    Args:
        df (pd.DataFrame): DataFrame in logn format (time, neurons)
        df_neurons (pd.DataFrame): DataFrame containing neuron meta information [i.e. one row per neurons and a column for group membership]
        group (str): Name of group to subset out
        group_col (str, optional): Column in DF neurons containing group information. Defaults to "group".
        neuron_col (str, optional): Column in DF neurons containing neuron identifiers. Defaults to "group".

    Returns:
        pd.DataFrame: A subset of df containing only neurons from the specified group
    """
    neurons = df_neurons.loc[lambda x: x[group_col] == group][neuron_col].unique()
    return df[[c for c in df.columns if c in neurons]]


def counts_to_rate(
    df: pd.DataFrame, bin_width: float, count_col: str = "counts"
) -> pd.DataFrame:
    """Convert spike counts to spike rate

    Args:
        df (pd.DataFrame): DF in long format
        bin_width (float): Bin width used for counts
        count_col (str, optional): Name of column containing counts. Defaults to "counts".

    Returns:
        pd.DataFrame: df with added 'spike_rate' column
    """
    return df.assign(spike_rate=lambda x: x[count_col].divide(bin_width))


def gaussian_smooth(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Apply a gaussian smoothing transformation to each column in a dataframe

    Args:
        df (pd.DataFrame): DF in wide format
        sigma (float): Sigma perameter for gaussian smoothing. Larger values increase kernel width. Returns original DF if sigma is 0.

    Returns:
        pd.DataFrame: DF with smoothed values
    """
    if sigma == 0:
        return df
    return df.apply(gaussian_filter1d, sigma=sigma)


def sort_by_activity_in_range(
    df: pd.DataFrame, t_start: float, t_stop: float, agg_func: Callable,
) -> pd.DataFrame:
    """Sort columns of a DataFrame in wide format by values in a given time range

    Args:
        df (pd.DataFrame): DataFrame in format (time, neurons)
        t_start (float): Start point of window
        t_stop (float): End point of window
        agg_func (Callable): Function to use to aggregate values for sorting (e.g. np.mean)

    Returns:
        pd.DataFrame: Input DataFrame with columns sorted by aggregation (larger to the left)
    """
    idx = (
        df.loc[(df.index <= t_start) & (df.index <= t_stop)]
        .apply(agg_func)
        .sort_values(ascending=False)
        .index.values
    )
    return df[idx]


def create_combined_col(
    df: pd.DataFrame, c1: str, c2: str, returned_colname: Optional[str] = None
) -> pd.DataFrame:
    """Create a column which is combination of existing columns

    Args:
        df (pd.DataFrame): Input DF
        c1 (str): Name of first column
        c2 (str): Name of second column
        returned_colname (Optional[str], optional): Name of created column. Defaults to 'c1_c2'.

    Returns:
        pd.DataFrame: Original df with appended column
    """
    if returned_colname is None:
        returned_colname = f"{c1}_{c2}"
    return df.assign(
        **{
            returned_colname: lambda x: x[c1]
            .astype(str)
            .str.cat(x[c2].astype(str), sep="_")
        }
    )
