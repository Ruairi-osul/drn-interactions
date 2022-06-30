from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, Dict
from drn_interactions.brain_state import get_state_piv
from typing import Optional, List
from drn_interactions.load import load_neurons_derived, load_derived_generic
from drn_interactions.spikes import SpikesHandler
from copy import deepcopy


def shuffle_both(X, y):
    """Shuffle the data"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def shuffle_X(X, y):
    """Shuffle X"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y


def shuffle_y(X, y):
    """Shuffle y"""
    perm = np.random.permutation(X.shape[0])
    return X, y[perm]


def shuffle_Xcols(X, y):
    """Shuffle Decorrelate X"""
    X = np.apply_along_axis(np.random.permutation, 0, X)
    return X, y


def shuffle_neither(X, y):
    """Shuffle neigher"""
    return X, y


class StateDecoder:
    def __init__(
        self,
        estimator,
        cv,
        shuffler=None,
        scoring="f1_macro",
        state_colname: str = "state",
    ):
        self.estimator = estimator
        self.cv = cv
        self.shuffler = shuffle_X if shuffler is None else shuffler
        self.scoring = scoring
        self.state_colname = state_colname
        self.pop_le = LabelEncoder()
        self.unit_le = LabelEncoder()

    def get_real_scores(self, spikes: pd.DataFrame, states: pd.Series) -> np.ndarray:
        y = self.pop_le.fit_transform(states)
        X = spikes.values
        self.pop_scores_ = self._run_single(X, y)
        return self.pop_scores_

    def _run_single(self, X, y, do_clone=False):
        estimator = self.estimator if not do_clone else clone(self.estimator)
        cv = self.cv if not do_clone else deepcopy(self.cv)
        if do_clone:
            cv.random_state = np.random.randint(0, 100000)
        return cross_val_score(estimator, X, y, cv=cv, scoring=self.scoring)

    def get_bootstrap_scores(
        self, spikes: pd.DataFrame, states: pd.Series, n_boot: int = 75
    ) -> np.ndarray:
        reps = []
        y = self.pop_le.fit_transform(states)
        X = spikes.values
        for _ in tqdm(range(n_boot)):
            X, y = self.shuffler(X, y)
            rep_scores = self._run_single(X, y, do_clone=True)
            reps.append(rep_scores)
        return np.vstack(reps)

    def get_unit_scores(self, spikes: pd.DataFrame, states: pd.Series) -> Dict:
        y = self.unit_le.fit_transform(states)
        X = spikes.values
        out = {}
        for i, n in enumerate(spikes.columns):
            out[n] = cross_val_score(
                self.estimator,
                X[:, i].reshape(-1, 1),
                y,
                cv=self.cv,
                scoring=self.scoring,
            )
        return out


class StateEncoder:
    def __init__(
        self, estimator, cv, shuffler=None, scoring="f1_macro", state_colname="state"
    ):
        self.estimator = estimator
        self.cv = cv
        self.shuffler = shuffler if shuffler is not None else shuffle_X
        self.scoring = scoring
        self.state_colname = state_colname

    def get_pop_scores(self, spikes: pd.DataFrame, states: pd.Series):
        neurons = spikes.columns
        df_all = spikes
        out = {}
        for neuron in tqdm(neurons):
            y = df_all[neuron].values
            X = df_all.drop(neuron, axis=1)
            out[neuron] = self._run_single(X, y, do_clone=True)

        return out

    def get_state_scores(self, spikes: pd.DataFrame, states: pd.Series):
        neurons = spikes.columns
        out = {}
        states = states.to_frame()
        for neuron in tqdm(neurons):
            y = spikes[neuron].values
            X = states
            out[neuron] = self._run_single(X, y, do_clone=True)
        return out

    def get_combined_scores(self, spikes: pd.DataFrame, states: pd.Series):
        neurons = spikes.columns
        df_all = spikes.join(states).copy()
        out = {}
        for neuron in tqdm(neurons):
            y = spikes[neuron].values
            X = df_all.drop(neuron, axis=1)
            out[neuron] = self._run_single(X, y, do_clone=True)
        return out

    def _run_single(self, X, y, do_clone=False):
        estimator = self.estimator if not do_clone else clone(self.estimator)
        cv = self.cv if not do_clone else deepcopy(self.cv)
        if do_clone:
            cv.random_state = np.random.randint(0, 100000)
        return cross_val_score(estimator, X, y, cv=cv, scoring=self.scoring)


class StateDecodeDataLoader:
    def __init__(
        self,
        session_name=None,
        block="pre",
        t_start=0,
        t_stop=1800,
        bin_width=1,
        state_colname="state",
    ):
        self.block = block
        self.t_start = t_start
        self.t_stop = t_stop
        self.session_name = session_name
        self.bin_width = bin_width
        self.state_colname = state_colname

    def set_session(self, session_name):
        self.session_name = session_name

    def load_states(self):
        all_sessions = load_derived_generic("eeg_states.csv").rename(
            columns={"cluster": self.state_colname}
        )
        return all_sessions[all_sessions["session_name"] == self.session_name].copy()

    def load_spikes(self):
        self.sh = SpikesHandler(
            block=self.block,
            bin_width=self.bin_width,
            session_names=[self.session_name],
            t_start=self.t_start,
            t_stop=self.t_stop,
        )
        return self.sh.binned_piv

    def __call__(self):
        return self.load_spikes(), self.load_states()


class StateDecodePreprocessor:
    def __init__(
        self,
        neuron_types: Optional[List[str]] = None,
        max_units: Optional[int] = None,
        thresh_empty: int = 0,
        shuffle: Optional[Callable] = None,
    ):
        self.neuron_types = neuron_types
        self.max_units = max_units
        self.thresh_empty = thresh_empty
        self.shuffle = shuffle

    def align_spikes(self, spikes, states):
        spikes = get_state_piv(spikes, states)
        states = spikes.pop("state")
        return spikes, states

    def subset_units(self, spikes):
        idx = np.random.randint(0, spikes.shape[0], self.max_units)
        return spikes.iloc[:, idx]

    def subset_neurontypes(self, spikes):
        neurons = load_neurons_derived().query("wf_3 in @self.neuron_types")
        idx = neurons.neuron_id.unique().tolist()
        return spikes[[c for c in spikes.columns if c in idx]]

    def remove_empty(self, spikes, states):
        sums = spikes.sum(axis=1)
        spikes = spikes[sums >= self.thresh_empty]
        states = states[sums >= self.thresh_empty]
        return spikes, states

    def __call__(self, spikes, states):
        spikes, states = self.align_spikes(spikes, states)
        if self.max_units is not None:
            spikes = self.subset_units(spikes)
        if self.neuron_types is not None:
            spikes = self.subset_neurontypes(spikes)
        spikes, states = self.remove_empty(spikes, states)
        if self.shuffle is not None:
            spikes, states = self.shuffle(spikes, states)
        return spikes.copy(), states.copy()
