from binit.bin import which_bin
import pandas as pd
from typing import Tuple
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter
import warnings
from drn_interactions.io import load_neurons_derived
from neurobox.long_transforms import get_closest_event
from .spikes import SpikesHandler


def get_state_piv(
    spikes, eeg, state_col="state", index_name="bin", eeg_time_col="timepoint_s"
):
    spikes = spikes.copy()
    return (
        spikes.reset_index()
        .assign(
            eeg_time=lambda x: which_bin(
                x[index_name].values,
                eeg[eeg_time_col].values,
                time_before=0,
                time_after=2,
            )
        )
        .merge(eeg, left_on="eeg_time", right_on=eeg_time_col)
        .set_index(index_name)[list(spikes.columns) + [state_col]]
    )


def get_state_long(spikes, eeg, index_name="bin", eeg_time_col="timepoint_s"):
    return (
        spikes.reset_index()
        .copy()
        .assign(
            eeg_bin=lambda x: which_bin(
                x[index_name].values,
                eeg[eeg_time_col].values,
                time_before=0,
                time_after=2,
            )
        )
        .merge(eeg, left_on="eeg_bin", right_on="timepoint_s")
        .drop("timepoint_s", axis=1)
    )


class STFTPreprocessor:
    def __init__(
        self,
        freq_z: bool = True,
        time_z: bool = True,
        gaussian_filter: bool = True,
        gaussian_sigma: Tuple[float, float] = (1.5, 1.5),
        t_start: float = 0,
        t_stop: float = 1800,
    ):
        self.freq_z = freq_z
        self.time_z = time_z
        self.gaussian_filter = gaussian_filter
        self.gaussian_sigma = gaussian_sigma
        self.t_start = t_start
        self.t_stop = t_stop

    def time_filter(self, df_fft):
        return df_fft.loc[
            lambda x: (x.index >= self.t_start) & (x.index <= self.t_stop)
        ]

    def freq_zscore(self, df_fft):
        return df_fft.apply(zscore)

    def time_zscore(self, df_fft):
        return df_fft.apply(zscore, axis=1)

    def gaussian(self, df_fft):
        vals = gaussian_filter(df_fft, sigma=self.gaussian_sigma)
        df_fft = pd.DataFrame(vals, index=df_fft.index, columns=df_fft.columns)
        return df_fft

    def transpose_ivert_freqs(self, df_fft):
        return df_fft.transpose().iloc[::-1]

    def __call__(self, df_fft):
        df_fft = self.time_filter(df_fft)
        if self.freq_z:
            df_fft = self.freq_zscore(df_fft)
        if self.time_z:
            df_fft = self.time_zscore(df_fft)
        if self.gaussian_filter:
            df_fft = self.gaussian(df_fft)
        df_fft = self.transpose_ivert_freqs(df_fft)
        return df_fft


class BrainStateUtils:
    """A container for a set of methods useful when working with brain state and spikes data"""

    def __init__(
        self,
        eeg_sampling_interval: float = 2,
        eeg_time_col: str = "timepoint_s",
        eeg_state_col: str = "state",
    ) -> None:
        self.eeg_sampling_interval = eeg_sampling_interval
        self.eeg_time_col = eeg_time_col
        self.eeg_state_col = eeg_state_col

    def _get_session_name(self, df: pd.DataFrame) -> pd.DataFrame:
        sessions = load_neurons_derived()[["neuron_id", "session_name"]]
        df = df.merge(sessions)
        return df

    def _align_data_to_states(
        self,
        df_data: pd.DataFrame,
        eeg_states: pd.DataFrame,
        df_data_time_col: str = "bin",
    ) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df_data = get_closest_event(
                df_data=df_data,
                df_events=eeg_states,
                time_after=self.eeg_sampling_interval,
                time_before=0,
                df_data_group_colname="session_name",
                df_events_group_colname="session_name",
                df_events_timestamp_col=self.eeg_time_col,
                df_data_time_col=df_data_time_col,
                returned_colname="eeg_bin",
            )
        df_data = df_data.drop("group", axis=1)
        return df_data

    def _get_state_from_aligned(
        self, df_data: pd.DataFrame, eeg_states: pd.DataFrame
    ) -> pd.DataFrame:
        df = df_data.merge(
            eeg_states[
                [self.eeg_time_col, self.eeg_state_col, "session_name"]
            ].drop_duplicates(),
            left_on=["session_name", "eeg_bin"],
            right_on=["session_name", self.eeg_time_col],
        )
        df = df.drop(self.eeg_time_col, axis=1)
        return df

    def align_spikes_state_long(
        self, sh: SpikesHandler, eeg_states: pd.DataFrame
    ) -> pd.DataFrame:
        spikes = sh.binned.copy()
        spikes = self._get_session_name(spikes)
        spikes = self._align_data_to_states(spikes, eeg_states, df_data_time_col="bin")
        df = self._get_state_from_aligned(spikes, eeg_states)
        return df

    def align_segmentedspikes_state(
        self, segmented_spikes: pd.DataFrame, eeg_states: pd.DataFrame
    ) -> pd.DataFrame:
        segmented_spikes = self._get_session_name(segmented_spikes)
        spikes = self._align_data_to_states(
            segmented_spikes, eeg_states, df_data_time_col="segment"
        )
        df = self._get_state_from_aligned(spikes, eeg_states)
        return df

    def align_eegraw_state(
        self, eeg_raw: pd.DataFrame, eeg_states: pd.DataFrame
    ) -> pd.DataFrame:
        spikes = self._align_data_to_states(
            eeg_raw, eeg_states, df_data_time_col="time"
        )
        df = self._get_state_from_aligned(spikes, eeg_states)
        return df
