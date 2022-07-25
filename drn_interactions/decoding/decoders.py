from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, Dict
from typing import Optional, List
from drn_interactions.io import load_neurons_derived, load_derived_generic, load_eeg
from drn_interactions.transforms import SpikesHandler
from copy import deepcopy
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from .shuffle import shuffle_X
from drn_interactions.config import Config, ExperimentInfo
from drn_interactions.transforms.brain_state_spikes import align_spikes_to_states_wide


class Decoder:
    def __init__(
        self,
        estimator,
        cv,
        shuffler=None,
        scoring="f1_macro",
    ):
        self.estimator = estimator
        self.cv = cv
        self.shuffler = shuffle_X if shuffler is None else shuffler
        self.scoring = scoring
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
