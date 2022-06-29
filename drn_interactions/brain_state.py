from binit.bin import which_bin
import numpy as np
import pandas as pd
import pingouin as pg
from typing import Dict, Optional, Any, Union
from drn_interactions.stats import p_adjust, mannwhitneyu_plusplus
from drn_interactions.plots import PAL_GREY_BLACK
import matplotlib.pyplot as plt
import seaborn as sns


def get_state_piv(
    spikes, eeg, state_col="state", index_name="bin", eeg_time_col="timepoint_s"
):
    spikes = spikes.copy()
    return (
        spikes.reset_index()
        .assign(
            eeg_time=lambda x: which_bin(
                x[index_name].values,
                eeg[eeg_time_col].values,
                time_before=0,
                time_after=2,
            )
        )
        .merge(eeg, left_on="eeg_time", right_on=eeg_time_col)
        .set_index(index_name)[list(spikes.columns) + [state_col]]
    )


def get_state_long(spikes, eeg, index_name="bin", eeg_time_col="timepoint_s"):
    return (
        spikes.reset_index()
        .copy()
        .assign(
            eeg_bin=lambda x: which_bin(
                x[index_name].values,
                eeg[eeg_time_col].values,
                time_before=0,
                time_after=2,
            )
        )
        .merge(eeg, left_on="eeg_bin", right_on="timepoint_s")
        .drop("timepoint_s", axis=1)
    )


class BSResonders:
    def __init__(
        self,
        df_value_col: str = "counts",
        df_neuron_col: str = "neuron_id",
        df_state_col: str = "state",
        clusters_neurontype_col: str = "wf_3",
        within_first: bool = True,
    ):
        self.df_value_col = df_value_col
        self.df_neuron_col = df_neuron_col
        self.df_state_col = df_state_col
        self.clusters_neurontype_col = clusters_neurontype_col
        self.within_first = within_first

    def get_anova(
        self, df: pd.DataFrame, clusters: Optional[pd.DataFrame] = None
    ) -> Dict:
        out = {}
        if clusters is None:
            out["anova"] = pg.rm_anova(
                data=df,
                dv=self.df_value_col,
                within=self.df_state_col,
                subject=self.df_neuron_col,
            ).round(3)
            out["contrasts"] = (
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
                .round(3)
            )
        else:
            df = df.merge(clusters)
            out["anova"] = pg.mixed_anova(
                data=df,
                dv=self.df_value_col,
                within=self.df_state_col,
                subject=self.df_neuron_col,
                between=self.clusters_neurontype_col,
            ).round(3)
            out["contrasts"] = (
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
                .round(3)
            )
        return out

    def get_responders(
        self,
        df: pd.DataFrame,
        abs_diff_thresh: float = 0,
    ) -> pd.DataFrame:
        self.responders = (
            df.groupby("neuron_id")
            .filter(
                lambda x: (len(x.query("state == 'sw'")) > 0)
                & (len(x.query("state == 'act'")) > 0)
            )
            .groupby("neuron_id")
            .apply(
                lambda x: mannwhitneyu_plusplus(
                    x.query("state == 'sw'")[self.df_value_col],
                    x.query("state == 'act'")[self.df_value_col],
                    names=["SW", "Act"],
                )
            )
            .assign(p=lambda x: p_adjust(x["p"]))
            .assign(sig=lambda x: np.where(x.p < 0.05, True, False))
            .round(3)
        )
        self.responders["sig"] = np.where(
            self.responders["Diff"].abs() < abs_diff_thresh,
            False,
            self.responders["sig"],
        )
        return self.responders

    def plot_responders(
        self,
        responders: pd.DataFrame = None,
        clusters: pd.DataFrame = None,
        bins: Any = "auto",
        cmap: Any = None,
        height: float = 3,
        aspect: float = 2,
    ) -> Union[plt.Axes, sns.FacetGrid]:
        """Plot Responders

        Args:
            responders (pd.DataFrame, optional): Output of get_responders method. Defaults to None.
            clusters (pd.DataFrame, optional): [neuron_id, wf_3]. Defaults to None.
            bins (Any, optional): To be passed to sns.histplot. Defaults to "auto".
            cmap (Any, optional): To be passed to sns.histplot. Defaults to None.
            height (float, optional): Height of the plot. Defaults to 3.
            aspect (float, optional): Aspect ratio of the plot. Defaults to 2.

        Returns:
            Union[plt.Axes, sns.FacetGrid]: Plot Object. Facet Grid if clusters specified.
        """
        if responders is None:
            responders = self.responders
        dfp = responders.reset_index().copy()
        if cmap is None:
            cmap = PAL_GREY_BLACK

        if clusters is None:
            _, ax = plt.subplots(figsize=(height, height * aspect))
            ax = sns.histplot(
                data=dfp,
                x="Diff",
                multiple="stack",
                alpha=1,
                bins=bins,
                ax=ax,
                hue="sig",
                hue_order=[True, False],
                palette=cmap,
                # edgecolor="black",
            )
            ax.set_xlabel("Spike Rate Change\n[Z Score]")
            ax.set_ylabel("Unit Counts")
            return ax
        else:
            dfp = dfp.merge(clusters)
            g = sns.FacetGrid(
                dfp,
                row=self.clusters_neurontype_col,
                sharey=False,
                height=height,
                aspect=aspect,
            ).map_dataframe(
                sns.histplot,
                x="Diff",
                hue="sig",
                multiple="stack",
                alpha=1,
                bins=bins,
                hue_order=[True, False],
                palette=cmap,
                # edgecolor="black",
            )
            g.set_axis_labels(x_var="Spike Rate Change\n[Z Score]", y_var="Unit Counts")
            return g


class BSRegularity:
    ...
