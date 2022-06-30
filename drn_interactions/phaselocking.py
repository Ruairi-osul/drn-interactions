import pandas as pd
from drn_interactions.spikes import SpikesHandler
from drn_interactions.load import (
    load_derived_generic,
    load_eeg_raw,
    load_neurons_derived,
)
from typing import Tuple
from scipy.signal import medfilt
from neurodsp.filt import filter_signal
from neurodsp.timefrequency import phase_by_time
import warnings
from neurobox.long_transforms import get_closest_event
import numpy as np
from astropy.stats import rayleightest, circmean, circstd
import astropy.units as u


def load_phaselock_data(
    block: str = "pre",
    state_quality_filter: bool = True,
    t_start: float = 0,
    t_stop: float = 1800,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_eeg = load_eeg_raw(block).rename(columns={"timepoint_s": "time"})
    eeg_states = load_derived_generic("eeg_states.csv").rename(
        columns={"cluster": "state"}
    )
    if state_quality_filter:
        eeg_states = eeg_states.query("quality == 'good'").copy()

    neurons = load_neurons_derived().merge(
        eeg_states[["session_name"]].drop_duplicates()
    )
    sessions = neurons[["session_name"]].drop_duplicates()

    df_eeg = df_eeg.merge(sessions).query("time > @t_start and time < @t_stop")

    eeg_states = eeg_states.merge(sessions).query(
        "timepoint_s > @t_start and timepoint_s < @t_stop"
    )
    sh = SpikesHandler(
        block="pre",
        bin_width=0.1,
        session_names=sessions["session_name"].unique().tolist(),
        t_start=t_start,
        t_stop=t_stop,
    )
    spikes = sh.spikes.copy()
    spikes = spikes.merge(neurons[["neuron_id", "session_name"]])
    spikes = spikes[["neuron_id", "session_name", "spiketimes"]]
    return neurons, spikes, df_eeg, eeg_states


def get_phase(df_eeg: pd.DataFrame) -> pd.DataFrame:
    # df_eeg["raw"] = df_eeg.groupby("session_name")["voltage"].transform(gaussian_filter1d, sigma=2)
    df_eeg["raw"] = df_eeg.groupby("session_name")["voltage"].transform(
        medfilt, kernel_size=5
    )

    df_eeg["delta"] = df_eeg.groupby("session_name")["raw"].transform(
        lambda x: filter_signal(
            x.values, fs=250, pass_type="bandpass", f_range=(0.2, 4)
        )
    )
    df_eeg["theta"] = df_eeg.groupby("session_name")["raw"].transform(
        lambda x: filter_signal(
            x.values, fs=250, pass_type="bandpass", f_range=(4.1, 8)
        )
    )

    df_eeg["delta_phase"] = df_eeg.groupby("session_name")["delta"].transform(
        lambda x: phase_by_time(x.values, fs=250)
    )
    df_eeg["theta_phase"] = df_eeg.groupby("session_name")["theta"].transform(
        lambda x: phase_by_time(x.values, fs=250)
    )
    return df_eeg


def align_spikes_to_phase(spikes: pd.DataFrame, df_eeg: pd.DataFrame) -> pd.DataFrame:
    df_eeg = df_eeg.dropna()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df_aligned = get_closest_event(
            df_data=spikes,
            df_events=df_eeg,
            time_after=0.1,
            time_before=0.1,
            df_data_group_colname="session_name",
            df_events_group_colname="session_name",
            df_events_timestamp_col="time",
            df_data_time_col="spiketimes",
            returned_colname="closest_bin",
        )
    df_aligned = df_aligned.dropna()
    df_aligned = df_aligned.merge(
        df_eeg,
        left_on=["session_name", "closest_bin"],
        right_on=["session_name", "time"],
    )
    return df_aligned


def circstats(x: np.ndarray) -> pd.Series:
    mean = circmean(x).to(u.degree).value
    std = circstd(x)
    p = rayleightest(x)
    return pd.Series({"mean": mean, "std": std, "p": p}).astype(float)
