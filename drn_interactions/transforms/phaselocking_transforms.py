from scipy.signal import medfilt
from neurodsp.filt import filter_signal
from neurodsp.timefrequency import phase_by_time
import warnings
from neurobox.long_transforms import get_closest_event
import numpy as np
from pingouin import circ_mean, circ_std, rayleightest
import pandas as pd


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


def circstats(x: np.ndarray) -> pd.Series:
    mean = circ_mean(x)
    std = circ_std(x)
    p = rayleightest(x)
    return pd.Series({"mean": mean, "std": std, "p": p}).astype(float)
