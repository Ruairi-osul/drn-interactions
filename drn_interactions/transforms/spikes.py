from .nbox_transforms import bin_spikes
from drn_interactions.io import load_neurons, load_spikes
from spiketimes.df.surrogates import shuffled_isi_spiketrains_by
import numpy as np
import pandas as pd


def concat_spikes_from_connsecutive_sessions(
    df_current: pd.DataFrame, df_next: pd.DataFrame, df_neurons: pd.DataFrame
) -> pd.DataFrame:
    """Concattenate DataFrames of spikes containing data from two consectutive sessions

    Args:
        df_current ([type]): DataFrame for current session
        df_next ([type]): DataFrame for next session
        df_neurons ([type]): Neurons DataFrame (from load_neurons())

    Returns:
        pd.DataFrame: The two DataFrames concattenated together with appropriate adjustments to spiketimes
    """
    df_neurons = df_neurons[["neuron_id", "cluster", "session_name", "group"]]
    df_current = df_current.merge(df_neurons)
    max_spikes = (
        df_current.groupby(["neuron_id", "session_name"], as_index=False)
        .spiketimes.max()
        .rename(columns={"spiketimes": "max_spike"})
    )
    df_next = (
        df_next.merge(max_spikes)
        .assign(spiketimes=lambda x: x.spiketimes.add(x.max_spike))
        .loc[lambda x: x.spiketimes >= x.max_spike]
        .drop("max_spike", axis=1)
        .merge(df_neurons)
    )
    return pd.concat([df_current, df_next])


class SpikesHandler:
    def __init__(
        self,
        block,
        bin_width,
        session_names=None,
        t_start=0,
        t_stop=None,
        shuffle=False,
    ):
        self.block = block
        self.bin_width = bin_width
        self.session_names = session_names
        self.t_start = t_start
        self.t_stop = t_stop
        self.shuffle = shuffle

        self.df_neurons = load_neurons()[["session_name", "neuron_id"]]
        if session_names is None:
            self.neuron_ids = self.df_neurons.neuron_id.unique()
        elif session_names == "random":
            self.session_names = (
                np.random.choice(self.df_neurons["session_name"].unique()),
            )
            self.neuron_ids = self.df_neurons.loc[
                lambda x: x.session_name.isin(self.session_names)
            ].neuron_id.unique()
        else:
            self.neuron_ids = self.df_neurons.loc[
                lambda x: x.session_name.isin(self.session_names)
            ].neuron_id.unique()

        if len(self.neuron_ids) == 0:
            raise ValueError("No Neurons")
        self._binned = None
        self._binned_piv = None
        self._spikes = None

    @property
    def spikes(self):
        if self._spikes is None:
            spikes = load_spikes(self.block).loc[
                lambda x: x.neuron_id.isin(self.neuron_ids)
            ]
            if self.t_stop is None:
                self.t_stop = spikes.spiketimes.max()
            spikes = spikes.loc[
                lambda x: (x.spiketimes >= self.t_start) & (x.spiketimes <= self.t_stop)
            ]
            if self.shuffle:
                spikes = shuffled_isi_spiketrains_by(
                    spikes, spiketimes_col="spiketimes", by_col="neuron_id", n=1
                )
            spikes = spikes.merge(self.df_neurons)
            self._spikes = spikes
        return self._spikes

    @property
    def binned(self):
        if self._binned is None:
            self._binned = bin_spikes(
                self.spikes,
                bin_width=self.bin_width,
                t_before=self.t_start,
                t_max=self.t_stop,
            )
        return self._binned

    @property
    def binned_piv(self):
        return self.binned.pivot(index="bin", columns="neuron_id", values="counts")


def get_population_train(df_piv):
    return df_piv.sum(axis=1).to_frame("population")


def pop_population_train(df_piv, exclude):
    df = df_piv.copy()
    neuron = df.pop(exclude).to_frame(name=exclude)
    pop_train = get_population_train(df)
    return neuron, pop_train


class SpikesHandlerMulti(SpikesHandler):
    @property
    def spikes(self):
        if self._spikes is None:
            shs = [
                SpikesHandler(
                    block=block,
                    bin_width=self.bin_width,
                    session_names=self.session_names,
                    t_start=self.t_start,
                    shuffle=self.shuffle,
                )
                for block in self.block
            ]
            spikes = shs[0].spikes
            for sh in shs[1:]:
                spikes2 = sh.spikes.loc[lambda x: x.spiketimes > 0].assign(
                    spiketimes=lambda x: x.spiketimes.add(spikes.spiketimes.max())
                )
                spikes = pd.concat([spikes, spikes2])
            self._spikes = spikes
            if self.t_stop is None:
                self.t_stop = self._spikes.spiketimes.max()
            self._spikes = self._spikes.loc[
                lambda x: (x.spiketimes >= self.t_start) & (x.spiketimes <= self.t_stop)
            ]
        return self._spikes
