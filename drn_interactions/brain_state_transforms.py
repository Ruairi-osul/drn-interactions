import numpy as np
from drn_interactions.load import load_neurons_derived
from drn_interactions.spikes import SpikesHandler
from binit.bin import which_bin
import warnings
from neurobox.long_transforms import get_closest_event


import pandas as pd


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

    @staticmethod
    def _get_session_name(df: pd.DataFrame) -> pd.DataFrame:
        sessions = load_neurons_derived()[["neuron_id", "session_name"]]
        df = df.merge(sessions)
        return df

    def _align_data_to_states(
        self,
        df_data: pd.DataFrame,
        eeg_states: pd.DataFrame,
        df_data_time_col: str = "bin",
    ) -> pd.DataFrame:
        df_data = self._get_session_name(df_data)
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
            eeg_states[[self.eeg_time_col, self.eeg_state_col, "session_name"]],
            left_on=["session_name", "eeg_bin"],
            right_on=["session_name", self.eeg_time_col],
        )
        df = df.drop(self.eeg_time_col, axis=1)
        return df

    def align_spikes_state_long(
        self, sh: SpikesHandler, eeg_states: pd.DataFrame
    ) -> pd.DataFrame:
        spikes = sh.binned.copy()
        spikes = self._align_data_to_states(spikes, eeg_states, df_data_time_col="bin")
        df = self._get_state_from_aligned(spikes, eeg_states)
        return df

    def align_segmentedspikes_state(
        self, segmented_spikes: pd.DataFrame, eeg_states: pd.DataFrame
    ) -> pd.DataFrame:
        spikes = self._align_data_to_states(
            segmented_spikes, eeg_states, df_data_time_col="segment"
        )
        df = self._get_state_from_aligned(spikes, eeg_states)
        return df
