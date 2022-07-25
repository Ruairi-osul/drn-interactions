import pandas as pd
import numpy as np
from drn_interactions.config import Config
from drn_interactions.transforms.spikes import SpikesHandlerMulti, SpikesHandler


class StateDecodeDataLoader:
    def __init__(
        self,
        session_name=None,
        block="pre",
        t_start=0,
        t_stop=1800,
        bin_width=1,
        state_colname="state",
        states_path=None,
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.state_colname = state_colname
        self.states_path = states_path or Config.derived_data_dir / "eeg_states.csv"

    def set_session(self, session_name):
        self.session_name = session_name

    def load_states(self):
        all_sessions = pd.read_csv(self.states_path)
        return all_sessions[all_sessions["session_name"] == self.session_name].copy()

    def load_spikes(self):
        session = (
            [self.session_name] if self.session_name is not None else self.session_name
        )
        self.sh = SpikesHandler(
            block=self.block,
            bin_width=self.bin_width,
            session_names=session,
            t_start=self.t_start,
            t_stop=self.t_stop,
        )
        return self.sh.binned_piv

    def __call__(self):
        return self.load_spikes(), self.load_states()


class FSDecodeDataLoader:
    def __init__(
        self,
        session_name=None,
        blocks=["base_shock", "post_base_shock"],
        t_start=-600,
        t_stop=1200,
        bin_width=1,
        state_colname="state",
    ):
        self.blocks = blocks
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.state_colname = state_colname
        self._spikes = None

    @property
    def spikes(self):
        if self._spikes is None:
            self._spikes = self.load_spikes()
        return self._spikes.copy()

    def set_session(self, session_name):
        self.session_name = session_name

    def load_states(self):
        arr = self.spikes.index.values
        block = np.where(arr < 0, "PRE", np.where(arr < 600, "SHOCK", "POST"))
        return pd.Series(block, index=arr)

    def load_spikes(self):
        session = (
            [self.session_name] if self.session_name is not None else self.session_name
        )
        self.sh = SpikesHandlerMulti(
            block=self.blocks,
            bin_width=self.bin_width,
            session_names=session,
            t_start=self.t_start,
            t_stop=self.t_stop,
        )
        return self.sh.binned_piv

    def __call__(self):
        return self.load_spikes(), self.load_states()
