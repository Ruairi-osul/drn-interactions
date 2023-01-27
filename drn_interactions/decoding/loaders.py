import pandas as pd
import numpy as np
from drn_interactions.config import Config
from drn_interactions.transforms.spikes import SpikesHandlerMulti, SpikesHandler
from drn_interactions.io import load_events
from drn_interactions.transforms.shock_transforms import ShockUtils


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
        blocks=("base_shock", "post_base_shock"),
        t_start=-600,
        t_stop=1200,
        bin_width=1,
    ):
        self.blocks = blocks
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
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


class FSFastDecodeDataLoader:
    def __init__(
        self,
        session_name=None,
        block="base_shock",
        t_start=0,
        t_stop=600,
        bin_width=0.01,
        window=(0.05, 0.2),
        state_colname="state",
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.state_colname = state_colname
        self.window = window
        self._spikes = None
        self.transformer = ShockUtils()

    def set_session(self, session_name):
        self.session_name = session_name

    def __call__(self):
        session = (
            [self.session_name] if self.session_name is not None else self.session_name
        )
        spikes = SpikesHandler(
            block="base_shock",
            bin_width=1,
            session_names=session,
            t_start=self.t_start,
            t_stop=self.t_stop,
        ).spikes
        events = load_events(block_name="base_shock").query(
            "event_s >= @self.t_start and event_s <= @self.t_stop"
        )
        df_binned = (
            self.transformer.aligned_binned_from_spikes(
                spikes,
                events,
                sessions=None,
                bin_width=self.bin_width,
            )
            .drop("event", axis=1)
            .set_index(["bin"])
        )
        states = np.where(
            (df_binned.index.values >= self.window[0])
            & (df_binned.index.values <= self.window[1]),
            "PRE",
            "POST",
        )
        # states = pd.DataFrame({"cos": })
        states = pd.Series(states, index=df_binned.index.values)
        return df_binned, states


class FSFastDecodeDataLoaderTwoWindows:
    def __init__(
        self,
        session_name=None,
        block="base_shock",
        t_start=0,
        t_stop=600,
        bin_width=0.01,
        window_1=(0.05, 0.2),
        window_2=(0.5, 0.8),
        state_colname="state",
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.state_colname = state_colname
        self.window_1 = window_1
        self.window_2 = window_2
        self._spikes = None
        self.transformer = ShockUtils()

    def set_session(self, session_name):
        self.session_name = session_name

    def __call__(self):
        session = (
            [self.session_name] if self.session_name is not None else self.session_name
        )
        spikes = SpikesHandler(
            block="base_shock",
            bin_width=1,
            session_names=session,
            t_start=self.t_start,
            t_stop=self.t_stop,
        ).spikes
        events = load_events(block_name="base_shock").query(
            "event_s >= @self.t_start and event_s <= @self.t_stop"
        )
        df_binned = (
            self.transformer.aligned_binned_from_spikes(
                spikes,
                events,
                sessions=None,
                bin_width=self.bin_width,
            )
            .drop("event", axis=1)
            .set_index(["bin"])
        )
        states = np.select(
            condlist=[
                (
                    (df_binned.index.values >= self.window_1[0])
                    & (df_binned.index.values <= self.window_1[1]),
                ),
                (
                    (df_binned.index.values >= self.window_2[0])
                    & (df_binned.index.values <= self.window_2[1]),
                ),
            ],
            choicelist=["PRE", "POST"],
            default="NA",
        ).flatten()
        idx = states != "NA"
        states = pd.Series(states, index=df_binned.index.values)
        states = states.loc[idx]
        df_binned = df_binned.loc[idx]

        return df_binned, states
