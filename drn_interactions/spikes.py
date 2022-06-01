from drn_interactions.transforms import bin_spikes
from drn_interactions.load import load_neurons, load_spikes
import numpy as np


class SpikesHandler:
    def __init__(self, block, bin_width, session_names=None, t_start=0, t_stop=None):
        self.block = block
        self.bin_width = bin_width
        self.session_names = session_names
        self.t_start = t_start
        self.t_stop = t_stop

        df_neurons = load_neurons()
        if session_names is None:
            self.neuron_ids = df_neurons.neuron_id.unique()
        elif session_names == "random":
            self.session_names = (
                np.random.choice(df_neurons["session_name"].unique()),
            )
            self.neuron_ids = df_neurons.loc[
                lambda x: x.session_name.isin(self.session_names)
            ].neuron_id.unique()
        else:
            self.neuron_ids = df_neurons.loc[
                lambda x: x.session_name.isin(self.session_names)
            ].neuron_id.unique()

        self._binned = None
        self._binned_piv = None
        self._spikes = None

    @property
    def spikes(self):
        if self._spikes is None:
            self._spikes = load_spikes(self.block).loc[
                lambda x: x.neuron_id.isin(self.neuron_ids)
            ]
            if self.t_stop is None:
                self.t_stop = self._spikes.spiketimes.max()
            self._spikes = self._spikes.loc[
                lambda x: (x.spiketimes >= self.t_start) & (x.spiketimes <= self.t_stop)
            ]
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

    @staticmethod
    def get_population_train(df):
        return df.sum(axis=1).to_frame("population")

    def pop_population_train(self, exclude):
        df = self.binned_piv.copy()
        neuron = df.pop(exclude).to_frame(name=exclude)
        pop_train = self.get_population_train(df)
        return neuron, pop_train

