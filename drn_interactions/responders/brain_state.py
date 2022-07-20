import pandas as pd
import pingouin as pg
from typing import Any, Optional, Dict, Tuple
import numpy as np
from drn_interactions.stats import p_adjust, mannwhitneyu_plusplus
from pingouin import circ_mean, circ_rayleigh
from scipy.stats import circvar


class SpikeRateResonders:
    def __init__(
        self,
        df_value_col: str = "counts",
        df_neuron_col: str = "neuron_id",
        df_state_col: str = "state",
        clusters_neurontype_col: str = "neuron_type",
        within_first: bool = True,
        round_output: Optional[int] = None,
        sw_state: str = "sw",
        act_state: str = "act",
    ):
        self.df_value_col = df_value_col
        self.df_neuron_col = df_neuron_col
        self.df_state_col = df_state_col
        self.clusters_neurontype_col = clusters_neurontype_col
        self.within_first = within_first
        self.round_output = round_output
        self.sw_state = sw_state
        self.act_state = act_state

    def _anova_no_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        anova = pg.rm_anova(
            data=df,
            dv=self.df_value_col,
            within=self.df_state_col,
            subject=self.df_neuron_col,
        )
        contrasts = (
            pg.pairwise_ttests(
                data=df,
                dv=self.df_value_col,
                within=self.df_state_col,
                subject=self.df_neuron_col,
                padjust="fdr_bh",
            )
            .assign(p=lambda x: p_adjust(x["p-unc"]))
            .assign(Sig=lambda x: np.where(x.p < 0.05, "*", ""))
            .drop(
                ["Paired", "Parametric", "alternative", "p-unc", "BF10", "hedges"],
                axis=1,
            )
        )
        return anova, contrasts

    def anova_with_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        anova = pg.mixed_anova(
            data=df,
            dv=self.df_value_col,
            within=self.df_state_col,
            subject=self.df_neuron_col,
            between=self.clusters_neurontype_col,
        )
        contrasts = (
            pg.pairwise_ttests(
                data=df,
                dv=self.df_value_col,
                within=self.df_state_col,
                subject=self.df_neuron_col,
                between=self.clusters_neurontype_col,
                padjust="fdr_bh",
                within_first=self.within_first,
            )
            .assign(p=lambda x: p_adjust(x["p-unc"]))
            .assign(Sig=lambda x: np.where(x.p < 0.05, "*", ""))
            .drop(
                ["Paired", "Parametric", "alternative", "p-unc", "BF10", "hedges"],
                axis=1,
            )
        )
        return anova, contrasts

    def get_anova(
        self, df: pd.DataFrame, fit_neuron_types: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not fit_neuron_types:
            anova, contrasts = self._anova_no_types(df)
        else:
            anova, contrasts = self.anova_with_types(df)

        if self.round_output is not None:
            anova = anova.round(self.round_output)
            contrasts = contrasts.round(self.round_output)
        return anova, contrasts

    def _must_have_both_states(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        return df.groupby(self.df_neuron_col).filter(
            lambda x: (len(x.query(f"{self.df_state_col} == '{self.sw_state}'")) > 2)
            & (len(x.query(f"{self.df_state_col} == '{self.act_state}'")) > 2)
        )

    def get_responders(
        self,
        df: pd.DataFrame,
        abs_diff_thresh: float = 0,
        sw_label: str = "sw",
        act_label: str = "act",
    ) -> pd.DataFrame:
        # df = self._must_have_both_states(df)
        responders = (
            df.groupby(self.df_neuron_col)
            .apply(
                lambda x: mannwhitneyu_plusplus(
                    x.query(f"{self.df_state_col} == '{self.sw_state}'")[
                        self.df_value_col
                    ],
                    x.query(f"{self.df_state_col} == '{self.act_state}'")[
                        self.df_value_col
                    ],
                    names=[sw_label, act_label],
                )
            )
            .assign(p=lambda x: p_adjust(x["p"]))
            .assign(sig=lambda x: np.where(x.p < 0.05, True, False))
        )
        responders["sig"] = np.where(
            responders["Diff"].abs() < abs_diff_thresh,
            False,
            responders["sig"],
        )
        if self.round_output is not None:
            responders = responders.round(self.round_output)
        return responders.reset_index()


class PhaseLockResponders:
    def __init__(
        self,
        fs: float = 250,
        round_output: Optional[int] = None,
        df_neuron_col: str = "neuron_id",
    ):
        self.fs = fs
        self.round_output = round_output
        self.df_neuron_col = df_neuron_col

    @staticmethod
    def _circstats(x: np.ndarray, fs=None) -> pd.Series:
        mean = circ_mean(x)
        z, p = circ_rayleigh(x, d=fs)
        var = circvar(x, low=-np.pi, high=np.pi)
        return pd.Series({"mean_angle": mean, "var": var, "stat": z, "p": p}).astype(
            float
        )

    def prefered_angles(self, df, phase_col: str = "phase"):
        df_res = df.groupby(self.df_neuron_col, as_index=False)[phase_col].apply(
            lambda x: self._circstats(x.values, fs=self.fs)
        )
        df_res = df_res.assign(p=lambda x: p_adjust(x["p"]))
        df_res = df_res.assign(sig=lambda x: np.where(x.p < 0.05, True, False))
        if self.round_output is not None:
            df_res = df_res.round(self.round_output)
        return df_res
