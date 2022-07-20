from drn_interactions.config import Config
from typing import Optional, Sequence, Tuple
from drn_interactions.io import load_eeg_raw
import pandas as pd
from scipy.signal import medfilt
from neurodsp.filt import filter_signal
from neurodsp.timefrequency import phase_by_time
from pingouin import circ_mean, circ_rayleigh
import numpy as np
import warnings
from neurobox.long_transforms import get_closest_event


class StateHandler:
    def __init__(
        self,
        states_path: Optional[str] = None,
        quality_to_include: Sequence[str] = ("good", "med", "poor"),
        states_column: str = "state",
        time_column: str = "timepoint_s",
        quality_column: str = "quality",
        session_column: str = "session_name",
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        session_names: Optional[Sequence[str]] = None,
    ):
        self.states_path = states_path or Config.derived_data_dir / "eeg_states.csv"
        self.quality_to_include = quality_to_include
        self.states_column = states_column
        self.time_column = time_column
        self.quality_column = quality_column
        self.session_column = session_column
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_names = session_names
        self._states_df = None

    def load_states_df(self):
        return pd.read_csv(self.states_path)

    @property
    def states_df(self):
        if self._states_df is None:
            df = self.load_states_df()
            self._states_df = self._preprocess_df(df)
        return self._states_df

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.t_start is not None:
            df = df[df[self.time_column] >= self.t_start]
        if self.t_stop is not None:
            df = df[df[self.time_column] <= self.t_stop]
        if self.session_names is not None:
            df = df[df[self.session_column].isin(self.session_names)]
        df = df[df[self.quality_column].isin(self.quality_to_include)]
        return df


class RawEEGHandler:
    def __init__(
        self,
        block: str,
        session_names: Optional[Sequence[str]] = None,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        time_column: str = "timepoint_s",
        session_column: str = "session_name",
        signal_column: str = "signal_name",
        signal_to_use: str = "eeg_occ",
        value_column: str = "voltage",
        medfilter_size: Optional[int] = 5,
        delta_frange: Tuple[float, float] = (0.2, 4),
        theta_frange: Tuple[float, float] = (4.1, 8),
        fs: float = 250,
    ):
        self.block = block
        self.session_names = session_names
        self.t_start = t_start
        self.t_stop = t_stop
        self.time_column = time_column
        self.session_column = session_column
        self.signal_column = signal_column
        self.signal_to_use = signal_to_use
        self.value_column = value_column
        self.medfilter_size = medfilter_size
        self.delta_frange = delta_frange
        self.theta_frange = theta_frange
        self.fs = fs
        self._raw_eeg_df = None

    def _load_raw_eeg_df(self):
        return load_eeg_raw(self.block)

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.t_start is not None:
            df = df[df[self.time_column] >= self.t_start]
        if self.t_stop is not None:
            df = df[df[self.time_column] <= self.t_stop]
        if self.session_names is not None:
            df = df[df[self.session_column].isin(self.session_names)]
        if self.medfilter_size is not None:
            df[self.value_column] = df.groupby([self.session_column])[
                self.value_column
            ].transform(medfilt, kernel_size=self.medfilter_size)

        df["delta"] = df.groupby([self.session_column])[self.value_column].transform(
            lambda x: filter_signal(
                x.values, pass_type="bandpass", f_range=self.delta_frange, fs=self.fs
            )
        )
        df["theta"] = df.groupby([self.session_column])[self.value_column].transform(
            lambda x: filter_signal(
                x.values, pass_type="bandpass", f_range=self.theta_frange, fs=self.fs
            )
        )
        df["delta_phase"] = df.groupby(self.session_column)["delta"].transform(
            lambda x: phase_by_time(x.values, fs=self.fs)
        )
        df["theta_phase"] = df.groupby(self.session_column)["theta"].transform(
            lambda x: phase_by_time(x.values, fs=self.fs)
        )
        return df

    @property
    def raw_eeg_df(self):
        if self._raw_eeg_df is None:
            df = self._load_raw_eeg_df()
            self._raw_eeg_df = self._preprocess_df(df)
        return self._raw_eeg_df


def align_to_state(
    df_state: pd.DataFrame,
    df_data: pd.DataFrame,
    df_data_time_col: str,
    time_after: float,
    df_state_time_col: str = "timepoint_s",
    df_state_session_col: str = "session_name",
    df_data_session_col: str = "session_name",
    df_state_state_col: str = "state",
    time_before: float = 0,
    returned_colname: str = "eeg_bin",
    dropna: bool = False,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df_aligned = get_closest_event(
            df_data=df_data,
            df_events=df_state,
            time_after=time_after,
            time_before=time_before,
            df_data_group_colname=df_data_session_col,
            df_events_group_colname=df_state_session_col,
            df_events_timestamp_col=df_state_time_col,
            df_data_time_col=df_data_time_col,
            returned_colname=returned_colname,
        )
    if dropna:
        df_aligned = df_aligned.dropna()
    df_aligned = df_aligned.merge(
        df_state[
            [df_state_time_col, df_state_session_col, df_state_state_col]
        ].drop_duplicates(),
        left_on=[df_data_session_col, returned_colname],
        right_on=[df_state_session_col, df_state_time_col],
    )
    df_aligned = df_aligned.drop(df_state_time_col, axis=1)
    return df_aligned


def circstats(x: np.ndarray) -> pd.Series:
    mean = circ_mean(x)
    z, p = circ_rayleigh(x)
    return pd.Series({"mean": mean, "stat": z, "p": p}).astype(float)
