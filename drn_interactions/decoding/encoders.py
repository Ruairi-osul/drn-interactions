from pathlib import Path
from sklearn.base import clone
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional
from copy import deepcopy
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from .shuffle import shuffle_X_df
from drn_interactions.config import Config


class StateEncoder:
    def __init__(
        self,
        estimator,
        cv,
        shuffler=None,
        scoring=r2_score,
        neuron_types_path: Optional[Path] = None,
        neuron_id_col: str = "neuron_id",
        neuron_type_col: str = "neuron_type",
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.cv = cv
        self.shuffler = shuffler if shuffler is not None else shuffle_X_df
        self.scoring = scoring
        self.verbose = verbose
        self.neuron_types_path = (
            neuron_types_path or Config.derived_data_dir / "neuron_types.csv"
        )
        self.df_neuron_types = pd.read_csv(self.neuron_types_path)
        self.neuron_id_col = neuron_id_col
        self.neuron_type_col = neuron_type_col
        self.neuron_types = self.df_neuron_types[self.neuron_type_col].dropna().unique()

    def run_pop(self, spikes, states, shuffle=False):
        neurons = spikes.columns
        df_all = spikes
        out = {}
        for neuron in tqdm(neurons, disable=not self.verbose):
            y = df_all[neuron].values
            X = df_all.drop(neuron, axis=1)
            out[neuron] = self._run_single(X, y, do_clone=True, shuffle=shuffle)
        self.pop_scores_ = out
        return self

    def run_state(self, spikes, states, shuffle=False):
        neurons = spikes.columns
        out = {}
        states = states.to_frame()
        for neuron in tqdm(neurons, disable=not self.verbose):
            y = spikes[neuron].values
            X = states
            out[neuron] = self._run_single(X, y, do_clone=True, shuffle=shuffle)
        self.state_scores_ = out
        return self

    def run_combined(self, spikes: pd.DataFrame, states: pd.Series, shuffle=False):
        neurons = spikes.columns
        df_all = spikes.join(states).copy()
        out = {}
        for neuron in tqdm(neurons, disable=not self.verbose):
            y = spikes[neuron].values
            X = df_all.drop(neuron, axis=1)
            out[neuron] = self._run_single(X, y, do_clone=True, shuffle=shuffle)
        self.combined_scores_ = out
        return self

    def run_dropout(
        self, spikes: pd.DataFrame, states: pd.Series, shuffle: bool = False
    ):
        neurons = spikes.columns
        df_all = spikes
        outer_dict = {}
        for neuron_type in tqdm(self.neuron_types, disable=not self.verbose):
            to_drop = (
                self.df_neuron_types[
                    self.df_neuron_types[self.neuron_type_col] == neuron_type
                ][self.neuron_id_col]
                .unique()
                .astype(str)
                .tolist()
            )
            preds = spikes.copy()[[c for c in spikes.columns if c not in to_drop]]
            inner_dict = {}
            for neuron in neurons:
                y = df_all[neuron].values
                if neuron in preds.columns:
                    X = preds.drop(neuron, axis=1)
                else:
                    X = preds
                cv_scores = self._run_single(X, y, do_clone=True, shuffle=shuffle)
                inner_dict[neuron] = np.mean(cv_scores["test_score"])
            outer_dict[neuron_type] = inner_dict
        self.dropout_scores_ = outer_dict

    def run_limit(
        self,
        spikes: pd.DataFrame,
        states: pd.Series,
        min_features: int = 1,
        max_features: Optional[int] = None,
        shuffle: bool = False,
    ):
        max_features = (spikes.shape[1] - 1) if max_features is None else max_features

        outer_dict = {}
        for neuron in tqdm(spikes.columns, disable=not self.verbose):
            inner_dict = {}
            for k_features in range(min_features, max_features + 1):
                estimator_k = clone(self.estimator)
                estimator_k.regressor.steps.insert(
                    -1,
                    (
                        "selector",
                        SelectKBest(k=k_features, score_func=f_regression),
                    ),
                )
                y = spikes[neuron].values
                X = spikes.drop(neuron, axis=1)
                try:
                    cv_scores = self._run_single(
                        X, y, do_clone=True, estimator=estimator_k, shuffle=shuffle
                    )
                    inner_dict[k_features] = np.mean(cv_scores["test_score"])
                except ValueError:
                    break
            outer_dict[neuron] = inner_dict
        self.limit_scores_ = outer_dict

    def get_limit_scores(self) -> Dict:
        return self.limit_scores_

    def _get_test_scores(self, scores_dict):
        return {
            neuron_id: scores["test_score"] for neuron_id, scores in scores_dict.items()
        }

    def _get_best_estimator(self, scores_dict):
        estimators = {}
        for neuron_id, scores in scores_dict.items():
            best_score_idx = np.argmax(scores["test_score"])
            estimator = scores["estimator"][best_score_idx]
            estimators[neuron_id] = estimator
        return estimators

    def _get_feature_importances(self, scores_dict):
        estimator_dict = self._get_best_estimator(scores_dict)
        out = {}
        for neuron_id, estimator in estimator_dict.items():
            all_features = estimator.feature_names_in_
            feature_importances = estimator.regressor_.steps[-1][
                -1
            ].feature_importances_
            out[neuron_id] = dict(zip(all_features, feature_importances))
        return out

    def _k_best_features(self, fi_dict, k):
        fi = list(fi_dict.values())
        features = np.array(list(fi_dict.keys()))
        idx = np.argsort(fi)[::-1]

        return features[idx[:k]]

    def get_pop_scores(
        self,
    ):
        return self._get_test_scores(self.pop_scores_)

    def get_state_scores(
        self,
    ):
        return self._get_test_scores(self.state_scores_)

    def get_combined_scores(
        self,
    ):
        return self._get_test_scores(self.combined_scores_)

    def get_dropout_scores(
        self,
    ):
        return self.dropout_scores_

    def _run_single(self, X, y, do_clone=False, estimator=None, shuffle=False):
        if estimator is None:
            estimator = self.estimator if not do_clone else clone(self.estimator)
        cv = self.cv if not do_clone else deepcopy(self.cv)
        if do_clone:
            cv.random_state = np.random.randint(0, 100000)
        estimators = []
        test_scores = []
        if shuffle:
            X, y = self.shuffler(X, y)
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            estimators.append(estimator)
            test_scores.append(self.scoring(y_test, pred))
        cv_results = dict(estimator=estimators, test_score=test_scores)
        return cv_results
