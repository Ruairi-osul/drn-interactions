from typing import Sequence
import pandas as pd
from drn_interactions.transforms.spikes import SpikesHandler, SpikesHandlerMulti
from drn_interactions.config import Config
from drn_interactions.transforms.brain_state_spikes import align_spikes_to_states_wide
import abc


class InteractionsLoader(abc.ABC):
    def __init__(
        self,
        session_name=None,
        block="pre",
        t_start=0,
        t_stop=1800,
        bin_width=1,
        shuffle=False,
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.shuffle = shuffle

    def set_session(self, session_name):
        self.session_name = session_name

    def __call__(self):
        ...


class SpontaneousActivityLoader(InteractionsLoader):
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
            shuffle=self.shuffle,
        )
        return self.sh.binned_piv

    def __call__(self):
        df = self.load_spikes()
        df.columns.name = "neuron_id"
        return df


class StateInteractionsLoader(InteractionsLoader):
    def __init__(
        self,
        state: str,
        session_name=None,
        block="pre",
        t_start=0,
        t_stop=1800,
        bin_width=1,
        states_path=None,
        state_colname="state",
        state_time_col="timepoint_s",
        shuffle=False,
    ):
        self.state = state
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.state_colname = state_colname
        self.states_path = states_path or Config.derived_data_dir / "eeg_states.csv"
        self.state_time_col = state_time_col
        self.shuffle = shuffle

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
            shuffle=self.shuffle,
        )
        return self.sh.binned_piv

    def load_states(self):
        all_sessions = pd.read_csv(self.states_path)
        return all_sessions[all_sessions["session_name"] == self.session_name].copy()

    def align_spikes(self, spikes, states):
        spikes = align_spikes_to_states_wide(spikes, states)
        states = spikes.pop("state")
        return spikes, states

    def align(self, spikes, states):
        return align_spikes_to_states_wide(
            spikes,
            states,
            state_col=self.state_colname,
            index_name="bin",
            eeg_time_col=self.state_time_col,
        )

    def subset_state(self, df):
        return df[df[self.state_colname] == self.state]

    def drop_state_col(self, df):
        return df.drop(self.state_colname, axis=1)

    def __call__(self):
        spikes = self.load_spikes()
        states = self.load_states()
        df = self.align(spikes, states)
        df = self.subset_state(df)
        df = self.drop_state_col(df)
        df.columns.name = "neuron_id"
        return df.copy()


class BaseShockSlowInteractionsLoader(InteractionsLoader):
    def __init__(
        self,
        block,
        session_name=None,
        t_start=-600,
        t_stop=1200,
        bin_width=1,
        shuffle=False,
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.shuffle = shuffle

    def load_spikes(self):
        session = (
            [self.session_name] if self.session_name is not None else self.session_name
        )
        self.sh = SpikesHandlerMulti(
            block=["base_shock", "post_base_shock"],
            bin_width=self.bin_width,
            session_names=session,
            t_start=self.t_start,
            t_stop=self.t_stop,
            shuffle=self.shuffle,
        )
        return self.sh.binned_piv

    def subset_block(self, df):
        idx = df.index
        if self.block == "pre":
            return df.loc[idx < 0]
        elif self.block == "base_shock":
            return df.loc[(idx >= 0) & (idx < 600)]
        elif self.block == "post_base_shock":
            return df.loc[idx >= 600]
        else:
            raise ValueError("Unknown block")

    def __call__(self):
        spikes = self.load_spikes()
        spikes = self.subset_block(spikes)
        spikes.columns.name = "neuron_id"
        return spikes.copy()


class FSFastInteractionsLoader(InteractionsLoader):
    ...
