from copy import deepcopy
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from .shuffle import shuffle_X
from drn_interactions.config import Config, ExperimentInfo
from drn_interactions.transforms.brain_state_spikes import align_spikes_to_states_wide
from typing import Callable, Optional, Sequence
from pathlib import Path
from .loaders import StateDecodeDataLoader
from .preprocessors import StateDecodePreprocessor
import pandas as pd
import numpy as np
from drn_interactions.io import load_eeg


class EEGDecodeLoader:
    def __init__(
        self,
        states_path: Optional[Path] = None,
        neuron_types_path: Optional[Path] = None,
        neuron_type_col: str = "neuron_type",
        block: str = "pre",
        fft_loader: Callable = load_eeg,
        min_freq: float = 0,
        max_freq: float = 8,
        sessions: Optional[Sequence] = None,
    ):
        self.states_path = states_path or Config.derived_data_dir / "eeg_states.csv"
        self.neuron_types_path = (
            neuron_types_path or Config.derived_data_dir / "neuron_types.csv"
        )
        self.neuron_type_col = neuron_type_col
        self.block = block
        self.fft_loader = fft_loader
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sessions = (
            sessions if sessions is not None else ExperimentInfo.eeg_sessions
        )

    @property
    def eeg_states(self):
        return pd.read_csv(self.states_path)

    @property
    def neurons(self):
        return pd.read_csv(self.neuron_types_path)

    @property
    def df_fft(self):
        return self.fft_loader(self.block).query(
            f"frequency > {self.min_freq} and frequency < {self.max_freq}"
        )

    def load_metadata(self):
        """Load data not bound to a specific session

        Returns:
            np.ndarray: sessions
            pd.DataFrame: neurons
        """
        return self.sessions, self.neurons

    def load_session_data(
        self, session_name, t_start=0, t_stop=1800, bin_width=1, thresh_empty=2
    ):
        """Load data from a specific session

        Returns:
            pd.DataFrame: spikes
            pd.DataFrame: states
            pd.DataFrame: df_fft
        """
        loader = StateDecodeDataLoader(
            session_name=session_name,
            block=self.block,
            t_start=t_start,
            t_stop=t_stop,
            bin_width=bin_width,
        )
        preprocessor = StateDecodePreprocessor(thresh_empty=thresh_empty)
        spikes, states = loader()
        spikes, states = preprocessor(spikes, states)
        spikes.columns = spikes.columns.map(str)
        df_fft = self.df_fft.query(
            "session_name == @session_name and timepoint_s <= @t_stop and timepoint_s >= @t_start"
        ).pivot(index="timepoint_s", columns="frequency", values="fft_value")
        return spikes, states, df_fft
