from .loaders import StateDecodeDataLoader
from .preprocessors import StateDecodePreprocessor, PreprocessingError
from .decoders import Decoder
from .encoders import StateEncoder
import numpy as np
import pandas as pd


class DecodeRunner:
    def __init__(
        self,
        loader: StateDecodeDataLoader,
        preprocessor: StateDecodePreprocessor,
        decoder: Decoder,
        nboot=None,
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        self.decoder = decoder
        self.nboot = nboot
        self._spikes = None
        self._states = None

    @property
    def spikes(self):
        if self._spikes is None:
            spikes, states = self.loader()
            spikes, states = self.preprocessor(spikes, states)
            self._spikes = spikes
            self._states = states
        return self._spikes

    @property
    def states(self):
        if self._states is None:
            spikes, states = self.loader()
            spikes, states = self.preprocessor(spikes, states)
            self._spikes = spikes
            self._states = states
        return self._states

    def pop_scores(self):
        return self.decoder.get_real_scores(self.spikes, self.states).mean()

    def shuffle_scores(self):
        return np.mean(
            self.decoder.get_bootstrap_scores(
                self.spikes, self.states, n_boot=self.nboot
            ),
            axis=1,
        ).mean()

    def unit_scores(self):
        unit_scores = self.decoder.get_unit_scores(self.spikes, self.states)
        return (
            pd.DataFrame(unit_scores)
            .mean()
            .to_frame("F1 Score")
            .reset_index()
            .rename(columns=dict(index="neuron_id"))
        )

    def run(
        self,
    ):
        pop_res = pd.DataFrame(
            dict(
                pop_true=self.pop_scores(),
                shuffle=self.shuffle_scores(),
            ),
            index=[0],
        )
        unit_res = self.unit_scores()
        return pop_res, unit_res

    def run_dropout(self, neuron_types=("SR", "SIR", "FF")):
        scores = np.empty(len(neuron_types))
        for i, to_drop in enumerate(neuron_types):
            nts = [nt for nt in neuron_types if nt != to_drop]
            self.preprocessor.neuron_types = nts
            self._spikes = None
            self._states = None
            try:
                scores[i] = self.pop_scores()
            except PreprocessingError:
                scores[i] = np.nan
        return pd.DataFrame({"dropped": neuron_types, "score": scores})

    def run_limit(self, n_min, n_max):
        n_units = np.arange(n_min, n_max)
        pop_scores = np.empty(len(n_units))
        for i, n_neurons in enumerate(n_units):
            self.preprocessor.exact_units = n_neurons
            self._spikes = None
            self._states = None
            try:
                pop_scores[i] = self.pop_scores()
            except PreprocessingError as e:
                pop_scores[i] = np.nan
            except ValueError as e:
                pop_scores[i] = np.nan
                print(str(e))
        return pd.DataFrame({"n_neurons": n_units, "pop_score": pop_scores})

    def run_multiple(self, sessions):
        pop_res = []
        unit_res = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            pop, unit = self.run()
            pop_res.append(pop.assign(session_name=session))
            unit_res.append(unit.assign(session_name=session))
        return pd.concat(pop_res).reset_index(drop=True), pd.concat(
            unit_res
        ).reset_index(drop=True)

    def run_multiple_limit(self, sessions, n_min, n_max):
        frames = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            frame = self.run_limit(n_min, n_max).assign(session_name=session)
            frames.append(frame)
        return pd.concat(frames).reset_index(drop=True)

    def run_multiple_dropout(self, sessions, neuron_types=("SR", "SIR", "FF")):
        frames = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            frame = self.run_dropout(neuron_types).assign(session_name=session)
            frames.append(frame)
        return pd.concat(frames).reset_index(drop=True)


class EncodeRunner:
    def __init__(
        self,
        loader: StateDecodeDataLoader,
        preprocessor: StateDecodePreprocessor,
        encoder: StateEncoder,
        nboot=None,
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.nboot = nboot
        self._spikes = None
        self._states = None

    @property
    def spikes(self):
        if self._spikes is None:
            spikes, states = self.loader()
            spikes, states = self.preprocessor(spikes, states)
            spikes.columns = spikes.columns.map(str)
            self._spikes = spikes
            self._states = states
        return self._spikes

    @property
    def states(self):
        if self._states is None:
            spikes, states = self.loader()
            spikes, states = self.preprocessor(spikes, states)
            self._spikes = spikes
            self._states = states
        return self._states

    def run_pop(self, shuffle=False):
        self.encoder.run_pop(self.spikes, self.states, shuffle=shuffle)
        pop = self.encoder.get_pop_scores()
        df_pop = (
            pd.DataFrame(pop)
            .mean()
            .to_frame("pop")
            .reset_index()
            .rename({"index": "neuron_id"}, axis=1)
        )
        return df_pop

    def run_state(self, shuffle=False):
        self.encoder.run_state(self.spikes, self.states, shuffle=shuffle)
        state = self.encoder.get_state_scores()
        df_state = (
            pd.DataFrame(state)
            .mean()
            .to_frame("state")
            .reset_index()
            .rename({"index": "neuron_id"}, axis=1)
        )
        return df_state

    def run_comb(self, shuffle=False):
        self.encoder.run_combined(self.spikes, self.states, shuffle=shuffle)
        comb = self.encoder.get_combined_scores()
        df_comb = (
            pd.DataFrame(comb)
            .mean()
            .to_frame("comb")
            .reset_index()
            .rename({"index": "neuron_id"}, axis=1)
        )
        return df_comb

    def run_limit(self, min_features, max_features, shuffle=False):
        self.encoder.run_limit(
            self.spikes,
            self.states,
            min_features=min_features,
            max_features=max_features,
            shuffle=shuffle,
        )
        limit = self.encoder.get_limit_scores()
        df_limit = pd.DataFrame(limit)
        df_limit = df_limit.reset_index().rename({"index": "n_best"}, axis=1)
        df_limit = df_limit.melt(
            id_vars="n_best", var_name="neuron_id", value_name="score"
        )
        return df_limit

    def run_dropout(self, shuffle=False):
        self.encoder.run_dropout(self.spikes, self.states, shuffle=shuffle)
        drop = self.encoder.get_dropout_scores()
        df_drop = pd.DataFrame(drop)
        df_drop = df_drop.reset_index().rename({"index": "neuron_id"}, axis=1)
        df_drop = df_drop.melt(
            id_vars="neuron_id",
            var_name=self.encoder.neuron_type_col,
            value_name="score",
        )
        return df_drop

    def run_multiple_pop(self, sessions, shuffle=False):
        frames = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            frame = self.run_pop(shuffle=shuffle).assign(session_name=session)
            frames.append(frame)
        return pd.concat(frames).reset_index(drop=True)

    def run_multiple_limit(self, sessions, min_features, max_features, shuffle=False):
        frames = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            frame = self.run_limit(
                min_features=min_features, max_features=max_features, shuffle=shuffle
            ).assign(session_name=session)
            frames.append(frame)
        return pd.concat(frames).reset_index(drop=True)

    def run_multiple_dropout(self, sessions, shuffle=False):
        frames = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            frame = self.run_dropout(shuffle=shuffle).assign(session_name=session)
            frames.append(frame, )
        return pd.concat(frames).reset_index(drop=True)

    def run_multiple_state(self, sessions, shuffle=False):
        frames = []
        for session in sessions:
            self._spikes = None
            self._states = None
            self.loader.set_session(session)
            frame = self.run_state(shuffle=shuffle).assign(session_name=session)
            frames.append(frame)
        return pd.concat(frames).reset_index(drop=True)
