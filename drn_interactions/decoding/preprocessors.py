from typing import Optional, Sequence, Callable
from pathlib import Path
from drn_interactions.config import Config
from drn_interactions.transforms.brain_state_spikes import align_spikes_to_states_wide
import pandas as pd
import numpy as np


class PreprocessingError(Exception):
    pass


class DecodePreprocessor:
    def __init__(
        self,
        thresh_empty: int = 0,
        neuron_types: Optional[Sequence[str]] = None,
        exact_units: Optional[int] = None,
        shuffle: Optional[Callable] = None,
        neuron_types_path: Optional[Path] = None,
    ):
        self.neuron_types = neuron_types
        self.exact_units = exact_units
        self.thresh_empty = thresh_empty
        self.shuffle = shuffle
        self.neuron_types_path = (
            neuron_types_path or Config.derived_data_dir / "neuron_types.csv"
        )

    def subset_units(self, spikes):
        if self.exact_units is not None:
            if spikes.shape[1] < self.exact_units:
                raise PreprocessingError("Not enough units")
        cols = np.random.choice(spikes.columns, size=self.exact_units, replace=False)
        return spikes.loc[:, cols]

    def subset_neurontypes(self, spikes):
        neurons = pd.read_csv(self.neuron_types_path).query(
            "neuron_type in @self.neuron_types"
        )
        idx = neurons.neuron_id.unique().tolist()
        return spikes[[c for c in spikes.columns if c in idx]]

    def remove_empty(self, spikes, states):
        sums = spikes.sum(axis=1)
        spikes = spikes[sums >= self.thresh_empty]
        states = states[sums >= self.thresh_empty]
        return spikes, states

    def __call__(self, spikes, states):
        if self.exact_units is not None:
            spikes = self.subset_units(spikes)
        if self.neuron_types is not None:
            spikes = self.subset_neurontypes(spikes)
        spikes, states = self.remove_empty(spikes, states)
        if self.shuffle is not None:
            spikes, states = self.shuffle(spikes, states)
        return spikes.copy(), states.copy()


class StateDecodePreprocessor(DecodePreprocessor):
    def align_spikes(self, spikes, states):
        spikes = align_spikes_to_states_wide(spikes, states)
        states = spikes.pop("state")
        return spikes, states

    def __call__(self, spikes, states):
        spikes, states = self.align_spikes(spikes, states)
        if self.exact_units is not None:
            spikes = self.subset_units(spikes)
        if self.neuron_types is not None:
            spikes = self.subset_neurontypes(spikes)
        spikes, states = self.remove_empty(spikes, states)
        if self.shuffle is not None:
            spikes, states = self.shuffle(spikes, states)
        return spikes.copy(), states.copy()
