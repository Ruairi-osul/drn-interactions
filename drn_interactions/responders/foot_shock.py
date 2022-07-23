from typing import Tuple, Optional, Dict, Sequence, Any
import pandas as pd
import numpy as np
import pingouin as pg
from drn_interactions.stats import p_adjust, mannwhitneyu_plusplus


class SpikeRateResponders:
    def __init__(
        self,
        df_value_col: str = "counts",
        df_neuron_col: str = "neuron_id",
        df_block_col: str = "block",
        neurontype_col: str = "neuron_type",
        within_first: bool = True,
        round_output: Optional[int] = None,
        anova_contrast_interaction: bool = True,
        pre_label: str = "1Pre",
        shock_label: str = "2Shock",
        post_label: str = "3Post",
        min_diff: float = 0.05,
    ):
        self.df_value_col = df_value_col
        self.df_neuron_col = df_neuron_col
        self.df_block_col = df_block_col
        self.neurontype_col = neurontype_col
        self.within_first = within_first
        self.round_output = round_output
        self.anova_contrast_interaction = anova_contrast_interaction
        self.pre_label = pre_label
        self.shock_label = shock_label
        self.post_label = post_label
        self.min_diff = min_diff

    def _anova_no_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        anova = pg.rm_anova(
            data=df,
            dv=self.df_value_col,
            within=self.df_block_col,
            subject=self.df_neuron_col,
        )
        contrasts = (
            pg.pairwise_ttests(
                data=df,
                dv=self.df_value_col,
                within=self.df_block_col,
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

    def _anova_with_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        anova = pg.mixed_anova(
            data=df,
            dv=self.df_value_col,
            within=self.df_block_col,
            subject=self.df_neuron_col,
            between=self.neurontype_col,
        )
        contrasts = (
            pg.pairwise_ttests(
                data=df,
                dv=self.df_value_col,
                within=self.df_block_col,
                subject=self.df_neuron_col,
                between=self.neurontype_col,
                padjust="fdr_bh",
                within_first=self.within_first,
                interaction=self.anova_contrast_interaction,
                return_desc=True,
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
            anova, contrasts = self._anova_with_types(df)

        if self.round_output is not None:
            anova = anova.round(self.round_output)
            contrasts = contrasts.round(self.round_output)
        return anova, contrasts

    def _round(self, df: pd.DataFrame) -> float:
        if self.round_output is not None:
            return df.round(self.round_output)
        return df

    def _get_sig(
        self,
        df: pd.DataFrame,
        p_col: str,
        diff_col="diff",
        sig_anova: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        sig = np.where(df[p_col] < 0.05, True, False)

        if self.min_diff is not None:
            mag_sig = np.where(df["diff"].abs() > self.min_diff, True, False)
            sig = sig & mag_sig
        if sig_anova is not None:
            sig = sig & df[self.df_neuron_col].isin(sig_anova)
        return sig

    def get_responders(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        pairwise = []
        anovas = []
        for neuron in df[self.df_neuron_col].unique():
            df_sub = df.loc[df[self.df_neuron_col] == neuron]
            df_anova = pg.anova(
                data=df_sub, dv=self.df_value_col, between=self.df_block_col
            )
            anovas.append(df_anova.assign(**{self.df_neuron_col: neuron}))
            df_pairwise = pg.pairwise_tukey(
                data=df_sub, dv=self.df_value_col, between=self.df_block_col
            )
            pairwise.append(df_pairwise.assign(**{self.df_neuron_col: neuron}))

        df_anovas = (
            pd.concat(anovas)
            .reset_index(drop=True)
            .assign(p_adj=lambda x: p_adjust(x["p-unc"].values))
            .pipe(self._round)
        )
        sig_neurons = df_anovas.query("p_adj < 0.05")[self.df_neuron_col].unique()

        unit_posthoc = (
            pd.concat(pairwise)
            .reset_index(drop=True)
            .assign(diff_inv=lambda x: x["diff"] * -1)
        )
        pre_to_shock = (
            unit_posthoc.query("A == @self.pre_label and B == @self.shock_label")
            .assign(p_adj=lambda x: p_adjust(x["p-tukey"].values))
            .assign(
                sig=lambda x: self._get_sig(x, "p_adj", "diff", sig_anova=sig_neurons)
            )
            .pipe(self._round)
            .copy()
        )
        shock_to_post = (
            unit_posthoc.query("A == @self.shock_label and B == @self.post_label")
            .assign(p_adj=lambda x: p_adjust(x["p-tukey"].values))
            .assign(
                sig=lambda x: self._get_sig(x, "p_adj", "diff", sig_anova=sig_neurons)
            )
            .pipe(self._round)
            .copy()
        )
        pre_to_post = (
            unit_posthoc.query("A == @self.pre_label and B == @self.post_label")
            .assign(p_adj=lambda x: p_adjust(x["p-tukey"].values))
            .assign(
                sig=lambda x: self._get_sig(x, "p_adj", "diff", sig_anova=sig_neurons)
            )
            .pipe(self._round)
            .copy()
        )
        return df_anovas, {
            "pre_to_shock": pre_to_shock,
            "shock_to_post": shock_to_post,
            "pre_to_post": pre_to_post,
        }


class AlignedResponders:
    def __init__(
        self,
        neuron_col: str = "neuron_id",
        value_col: str = "counts",
        bin_col: str = "bin",
        window_col: str = "window",
        neuron_type_col: str = "neuron_type",
        within_first: bool = True,
        anova_contrast_interaction: bool = True,
        round_output: Optional[int] = None,
        shock_label: str = "post",
        pre_label: str = "pre",
    ):
        self.neuron_col = neuron_col
        self.value_col = value_col
        self.bin_col = bin_col
        self.neuron_type_col = neuron_type_col
        self.window_col = window_col
        self.within_first = within_first
        self.anova_contrast_interaction = anova_contrast_interaction
        self.round_output = round_output
        self.shock_label = shock_label
        self.pre_label = pre_label

    def _anova_no_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        anova = pg.rm_anova(
            data=df,
            dv=self.value_col,
            within=self.window_col,
            subject=self.neuron_col,
        )
        contrasts = (
            pg.pairwise_ttests(
                data=df,
                dv=self.value_col,
                within=self.window_col,
                subject=self.neuron_col,
                padjust="fdr_bh",
                return_desc=True
            )
            .assign(p=lambda x: p_adjust(x["p-unc"]))
            .assign(Sig=lambda x: np.where(x.p < 0.05, "*", ""))
            .drop(
                ["Paired", "Parametric", "alternative", "p-unc", "BF10", "hedges"],
                axis=1,
            )
        )
        return anova, contrasts

    def _anova_with_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        anova = pg.mixed_anova(
            data=df,
            dv=self.value_col,
            within=self.window_col,
            subject=self.neuron_col,
            between=self.neuron_type_col,

        )
        contrasts = (
            pg.pairwise_ttests(
                data=df,
                dv=self.value_col,
                within=self.window_col,
                subject=self.neuron_col,
                between=self.neuron_type_col,
                padjust="fdr_bh",
                within_first=self.within_first,
                interaction=self.anova_contrast_interaction,
                return_desc=True

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
            anova, contrasts = self._anova_with_types(df)

        if self.round_output is not None:
            anova = anova.round(self.round_output)
            contrasts = contrasts.round(self.round_output)
        return anova, contrasts

    def get_responders(
        self,
        df: pd.DataFrame,
        abs_diff_thresh: float = 0,
    ) -> pd.DataFrame:
        responders = (
            df.groupby(self.neuron_col)
            .apply(
                lambda x: mannwhitneyu_plusplus(
                    x.query(f"{self.window_col} == '{self.pre_label}'")[self.value_col],
                    x.query(f"{self.window_col} == '{self.shock_label}'")[
                        self.value_col
                    ],
                    names=[self.pre_label, self.shock_label],
                )
            )
            .assign(p=lambda x: p_adjust(x["p"]))
            .assign(sig=lambda x: np.where(x.p < 0.05, True, False))
        )
        responders["sig"] = np.where(
            responders["Diff"].abs() <= abs_diff_thresh,
            False,
            responders["sig"],
        )
        if self.round_output is not None:
            responders = responders.round(self.round_output)
        return responders.reset_index()
