from typing import Sequence
import numpy as np
import pingouin as pg
from .utils import corr_df_to_tidy, zero_diag_df
import abc
import pandas as pd


class PairwiseMetric(abc.ABC):
    value_col = "value"

    def __init__(
        self,
        zero_diag: bool = True,
        remove_duplicate_combs: bool = True,
        rectify: bool = False,
        shuffle: bool = False,
    ) -> None:
        self.zero_diag = zero_diag
        self.remove_duplicate_combs = remove_duplicate_combs
        self.rectify = rectify
        self.shuffle = shuffle

    @abc.abstractmethod
    def __call__(self, df):
        ...

    def get_adjacency_matrix(self):
        return self.A_

    def get_corr_df(
        self,
    ):
        if self.remove_duplicate_combs:
            return self._remove_duplicate_combs(self.df_corr_.copy())
        return self.df_corr_

    def shuffle_interactions(self):
        self.df_corr_[self.value_col] = np.random.permutation(
            self.df_corr_[self.value_col].values
        )
        self.A_ = self.df_corr_.pivot(
            index="neuron_1", columns="neuron_2", values=self.value_col
        )

    @staticmethod
    def _remove_duplicate_combs(df):
        not_duplicate_comb = pd.DataFrame(
            np.sort(df[["neuron_1", "neuron_2"]].values, 1)
        ).duplicated()
        df = df[not_duplicate_comb].copy()

    @staticmethod
    def _rectify(df_corr):
        vals = df_corr.values
        vals[vals < 0] = 0
        return pd.DataFrame(vals, index=df_corr.index, columns=df_corr.columns)


class PairwiseCorr(PairwiseMetric):
    value_col = "corr"

    def fit(self, df_piv, y=None):
        self.X_ = df_piv
        self.features_in_ = df_piv.columns.values

        self.A_ = self.X_.corr()

        self.df_corr_ = corr_df_to_tidy(
            self.A_,
            value_name=PairwiseCorr.value_col,
        )
        if self.shuffle:
            self.shuffle_interactions()
        if self.zero_diag:
            self.A_ = zero_diag_df(self.A_)

        if self.rectify:
            self.A_ = self._rectify(self.A_)

        self.df_corr_ = corr_df_to_tidy(
            self.A_,
            value_name=PairwiseCorr.value_col,
        )

        return self

    def __call__(self, df_piv, y=None):
        self.fit(df_piv, y)
        return self.get_corr_df()


class PairwisePartialCorr(PairwiseMetric):
    value_col = "pcorr"

    def fit(self, df_piv, y=None):
        self.X_ = df_piv
        self.features_in_ = df_piv.columns.values

        self.A_ = self.X_.pcorr()
        self.df_corr_ = corr_df_to_tidy(
            self.A_,
            value_name=PairwisePartialCorr.value_col,
        )
        if self.shuffle:
            self.shuffle_interactions()
        if self.zero_diag:
            self.A_ = zero_diag_df(self.A_)

        if self.rectify:
            self.A_ = self._rectify(self.A_)

        self.df_corr_ = corr_df_to_tidy(
            self.A_,
            value_name=PairwiseCorr.value_col,
        )
        return self

    def __call__(self, df_piv, y=None):
        self.fit(df_piv, y)
        return self.get_corr_df()


# class PairwiseRunner:
#     def __init__(self, corr_calculators: Sequence[PairwiseMetric]):
#         self.corr_calculators = corr_calculators
#         ...
