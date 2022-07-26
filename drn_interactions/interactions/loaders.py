from drn_interactions.transforms.spikes import SpikesHandler
from drn_interactions.config import Config
import abc


class InteractionsLoader(abc.ABC):
    def __init__(
        self,
        session_name=None,
        block="pre",
        t_start=0,
        t_stop=1800,
        bin_width=1,
        states_path=None,
        shuffle=False,
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.states_path = states_path or Config.derived_data_dir / "eeg_states.csv"
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
        return self.load_spikes()


class StateInteractionsLoader(InteractionsLoader):
    ...


class FSSlowInteractionsLoader(InteractionsLoader):
    ...


class FSFastInteractionsLoader(InteractionsLoader):
    ...
