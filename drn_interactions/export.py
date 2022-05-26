"""
Functions for dealing with the data export from the database to .parquet files
"""
from ephys_queries.queries import select_discrete_data, select_stft, select_waveforms
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.schema import MetaData
from drn_interactions.load import get_group_names
from ephys_queries import select_neurons, select_spike_times
from spiketimes.df.statistics import (
    mean_firing_rate_by,
    cv2_isi_by,
)
import pandas as pd
from pathlib import Path
from typing import Optional
import math
import joblib
from itertools import combinations


def load_neurons(engine: Engine, metadata: MetaData) -> pd.DataFrame:
    """Loads neurons from the database
    """
    group_names = get_group_names()
    neurons = select_neurons(engine, metadata, group_names=group_names)
    return neurons


def load_eeg(
    engine: Engine,
    metadata: MetaData,
    signal_name: str = "eeg_occ",
    t_before: int = 0,
    align_to_block=False,
) -> pd.DataFrame:
    group_names = get_group_names()
    try:
        neurons = select_stft(
            engine,
            metadata,
            group_names=group_names,
            signal_names=[signal_name],
            align_to_block=align_to_block,
            t_before=t_before,
        )
    except IndexError:
        neurons = pd.DataFrame()
    return neurons


def load_spiketimes(
    engine: Engine,
    metadata: MetaData,
    block_name: str,
    clusters: Optional[pd.DataFrame] = None,
    t_before: int = 0,
    align_to_block: bool = False,
) -> pd.DataFrame:
    """Loads spiketimes from the database

    Args:
        block_name (str): The name of the block to be loaded
        clusters (str): DataFrame with columns neuron_id and cluster
    
    Returns:
        pd.DataFrame: df of spiketimes
    """
    group_names = get_group_names()
    fs = 30000
    spiketimes = select_spike_times(
        engine,
        metadata,
        block_name=block_name,
        group_names=group_names,
        t_before=t_before,
        align_to_block=align_to_block,
    ).assign(spiketimes=lambda x: x["spike_time_samples"].divide(fs))
    if clusters is not None:
        spiketimes = spiketimes.merge(clusters[["neuron_id", "cluster"]], how="left")
    return spiketimes


def load_waveforms(
    engine: Engine, metadata: MetaData,
) -> pd.DataFrame:
    group_names = get_group_names()
    return select_waveforms(engine, metadata, group_names=group_names)


def load_events(
    engine: Engine, metadata: MetaData, block_name: str, align_to_block: bool = False,
) -> pd.DataFrame:
    fs = 30000
    group_names = get_group_names()
    df_events = select_discrete_data(
        engine,
        metadata,
        group_names=group_names,
        block_name=block_name,
        align_to_block=align_to_block,
    ).assign(event_s=lambda x: x["timepoint_sample"].divide(fs))
    return df_events


def classify_clusters(
    engine, metadata, PATH_TO_CLF: Path, PATH_TO_SCALER: Path, PATH_TO_LE: Path,
) -> pd.DataFrame:
    """Classify neurons into clusters based on previous experiment

    Args:
        df_spikes (pd.DataFrame): DataFrame containing spiketimes
        df_neurons (pd.DataFrame): DataFrame containing neurons
        PATH_TO_CLF (Path): Path to clf from previous experiment
        PATH_TO_SCALER (Path): Path to scaler fit to previous data
        PATH_TO_LE (Path): Path to label encoder for output of classifer

    Returns:
        pd.DataFrame: DataFrame with one row for each neuron containing column 'cluster' denoting neuron type
    """
    df_spikes = load_spiketimes(engine, metadata, block_name="pre")
    df_neurons = load_neurons(engine, metadata).rename(columns={"id": "neuron_id"})
    mfr = mean_firing_rate_by(
        df_spikes, spiketimes_col="spiketimes", spiketrain_col="neuron_id"
    )
    cv_isi = cv2_isi_by(
        df_spikes, spiketimes_col="spiketimes", spiketrain_col="neuron_id"
    )
    df = pd.merge(mfr, cv_isi).merge(df_neurons)
    df_sub = df.loc[lambda x: x.mean_firing_rate >= 0.2]
    mod = joblib.load(PATH_TO_CLF)
    scaler = joblib.load(PATH_TO_SCALER)
    le = joblib.load(PATH_TO_LE)
    X = scaler.transform(df_sub[["mean_firing_rate", "cv2_isi"]].values)
    df_sub["cluster"] = le.inverse_transform(mod.predict(X))
    out = df.merge(df_sub[["neuron_id", "cluster"]], how="left")
    return out


def calculate_distance(engine: Engine, metadata: MetaData):
    def _distance_between_chans(ch1, ch2):
        """
            Calculate distances between two channels on a cambridge neurotech 32 channel P series probe.
            
            Electrode spec:
                2 shanks 250um appart
                Each shank has 16 channels in two columns of 8, spaced 22.5um appart
                Contacts are placed 25um above eachother
            """
        # Shank
        shank_1 = 1 if ch1 <= 15 else 2
        shank_2 = 1 if ch2 <= 15 else 2
        width = 250 if shank_1 != shank_2 else 0

        # Column
        col_1 = 1 if ch1 % 2 == 0 else 2
        col_2 = 1 if ch2 % 2 == 0 else 2
        width = 22.5 if (col_1 != col_2) and (width == 0) else width

        #
        ch1t = ch1 - 16 if ch1 > 15 else ch1
        ch2t = ch2 - 16 if ch2 > 15 else ch2
        height = abs(ch1t - ch2t) * 25

        return math.hypot(height, width) if width else height

    df_neurons = load_neurons(engine, metadata)
    dfs = []
    for session_name in df_neurons.session_name.unique():
        neurons = df_neurons.loc[lambda x: x.session_name == session_name]
        combs = combinations(neurons.id, r=2)
        c1s, c2s = [], []
        for comb in combs:
            c1s.append(comb[0])
            c2s.append(comb[1])
        by_comb = pd.DataFrame({"neuron1": c1s, "neuron2": c2s})
        by_comb = (
            by_comb.merge(neurons[["id", "channel"]], left_on="neuron1", right_on="id")
            .drop("id", axis=1)
            .rename(columns={"channel": "channel_n1"})
        )
        by_comb = (
            by_comb.merge(neurons[["id", "channel"]], left_on="neuron2", right_on="id")
            .drop("id", axis=1)
            .rename(columns={"channel": "channel_n2"})
        )
        by_comb["distance"] = by_comb.apply(
            lambda x: _distance_between_chans(x.channel_n1, x.channel_n2), axis=1
        )
        by_comb["session_name"] = session_name
        dfs.append(by_comb)
    return pd.concat(dfs)
