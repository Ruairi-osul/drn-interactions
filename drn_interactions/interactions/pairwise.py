from typing import Callable, Optional, Sequence
import numpy as np
import pingouin as pg
from .utils import corr_df_to_tidy, zero_diag_df, shuffle_df_corr
import abc
import pandas as pd


class PairwiseMetric(abc.ABC):
    value_col = "value"
    corr_calculator: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    def __init__(
        self,
        remove_self_interactions: bool = True,
        remove_duplicate_combs: bool = True,
        rectify: bool = False,
        shuffle: bool = False,
        neuron_col: str = "neuron_id",
        comb_col_name: str = "comb",
        add_x: Optional[float] = None,
        dropna: bool = True,
    ) -> None:
        self.remove_self_interactions = remove_self_interactions
        self.remove_duplicate_combs = remove_duplicate_combs
        self.rectify = rectify
        self.shuffle = shuffle
        self.neuron_col = neuron_col
        self.add_x = add_x
        self.comb_col_name = comb_col_name
        self.dropna = dropna

    def _make_edge_df(self, df_corr):
        df_edge = (
            df_corr.copy()
            .reset_index()
            .rename(columns={self.neuron_col: "neuron_1"})
            .melt(id_vars="neuron_1", var_name="neuron_2", value_name=self.value_col)
        )
        if self.remove_duplicate_combs:
            df_edge = self._remove_duplicate_combs(df_edge)
        df_edge[self.comb_col_name] = df_edge.apply(
            lambda x: sorted(list({x["neuron_1"], x["neuron_2"]})), axis=1
        )
        df_edge[self.comb_col_name] = df_edge[self.comb_col_name].astype(str)
        return df_edge.copy()

    def get_adjacency_df(self):
        return self.df_corr_

    def get_edge_df(self):
        return self._make_edge_df(self.df_corr_)

    def _shuffle_interactions(self):
        self.df_corr_ = shuffle_df_corr(self.df_corr_)

    def _remove_self_interactions(self):
        self.df_corr_ = zero_diag_df(self.df_corr_)

    def _dropna(self):
        self.df_corr_ = self.df_corr_.dropna(axis=1).dropna(thresh=5, axis=0)

    def fit(self, df, y=None):
        self.df_corr_ = self.corr_calculator(df)
        if self.shuffle:
            self._shuffle_interactions()
        if self.add_x:
            self._add_x()
        if self.remove_self_interactions:
            self._remove_self_interactions()
        if self.rectify:
            self._rectify()
        if self.dropna:
            self._dropna()
        return self

    def _add_x(self):
        self.df_corr_ = self.df_corr_ + self.add_x

    @staticmethod
    def _remove_duplicate_combs(df):
        not_duplicate_comb = pd.DataFrame(
            np.sort(df[["neuron_1", "neuron_2"]].values, 1)
        ).duplicated()
        return df[not_duplicate_comb].copy()

    def _rectify(self):
        vals = self.df_corr_.values
        vals[vals < 0] = 0
        self.df_corr_ = pd.DataFrame(
            vals, index=self.df_corr_.index, columns=self.df_corr_.columns
        )

    def __call__(self, df_piv, y=None):
        self.fit(df_piv, y)
        return self.get_adjacency_df()


class PairwiseCorr(PairwiseMetric):
    value_col = "corr"

    def corr_calculator(self, df):
        return df.corr()


class PairwisePartialCorr(PairwiseMetric):
    value_col = "pcorr"

    def corr_calculator(self, df):
        return df.pcorr()
