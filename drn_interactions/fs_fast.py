from optparse import Option
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import zscore, variation
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from seaborn.axisgrid import FacetGrid
from .shock_transforms import ShockUtils
from .stats import mannwhitneyu_plusplus, p_adjust
from .plots import PAL_GREY_BLACK


class ShortTSResponders:
    def __init__(self, window: Tuple[float, float] = (0.05, 0.3)):
        """Define Foot Shock Responses

        Args:
            window (Tuple[float, float], optional): Period around which foot shock responses are considered. Defaults to (0.1, 0.3).
        """
        self.window = window
        self.transformer = ShockUtils()

    def get_responders(self, df: pd.DataFrame):
        ...


class ShortTsAnova(ShortTSResponders):
    """Across all units, was the mean activity different inside and outside the window?
    """

    def get_responders(
        self,
        df_binned_aligned: pd.DataFrame,
        clusters: pd.DataFrame = None,
        z: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run Anova on Pre v Post. Optionally include neuron type and its interaction

        Args:
            df_binned_aligned (pd.DataFrame): bin as index, neuron_id as columns, values as spike counts
            clusters (pd.DataFrame, optional): N[neuron_id, wf_3]. Defaults to None.
            z (bool, optional): Whether to Z score the average trace before funning the statistics. Defaults to False.

        Returns:
            pd.DataFrame: Anova Table
            pd.DataFrame: Contrast Table
        """
        avg_trace = self.transformer.average_trace_from_aligned_binned(
            df_binned_aligned
        )
        df = avg_trace.reset_index().melt(id_vars="bin", var_name="neuron_id")
        if z:
            df["value"] = df.groupby("neuron_id")["value"].transform(zscore)
        df["prepost"] = np.where(
            (df["bin"] < self.window[0]) | (df["bin"] > self.window[1]), "pre", "post"
        )
        if clusters is None:
            self.anova = pg.rm_anova(
                data=df, dv="value", within="prepost", subject="neuron_id"
            )
            self.contrasts = pg.pairwise_ttests(
                data=df, dv="value", within="prepost", subject="neuron_id"
            )
        else:
            df = df.merge(clusters)
            self.anova = pg.mixed_anova(
                data=df,
                dv="value",
                within="prepost",
                subject="neuron_id",
                between="wf_3",
            ).round(3)
            self.contrasts = (
                pg.pairwise_ttests(
                    data=df,
                    dv="value",
                    within="prepost",
                    subject="neuron_id",
                    between="wf_3",
                    padjust="fdr_bh",
                    interaction=True,
                    # within_first=False,
                )
                .query(
                    "prepost != 'pre' and Contrast != 'wf_3'"
                )  # extract only meaningful contrasts
                .assign(p=lambda x: p_adjust(x["p-unc"]))
                .round(3)
                .drop(
                    [
                        "Parametric",
                        "Paired",
                        "p-adjust",
                        "BF10",
                        "hedges",
                        "alternative",
                    ],
                    axis=1,
                )
                .assign(Sig=lambda x: np.where(x["p"] < 0.05, "*", ""))
            )
        return self.anova, self.contrasts


class ShortTsAvg(ShortTSResponders):
    """Define Foot Shock Response of Each Neuron Based on its Average Trace
    """

    def get_responders(
        self, df_binned_aligned: pd.DataFrame, z: bool = False
    ) -> pd.DataFrame:
        """Get response of each neuron pre v post using MannWhiteney U tests.

        Args:
            df_binned_aligned (pd.DataFrame): Bins as index, neuron_id as columns, spike counts as values
            z (bool, optional): Whether to Z score prior to computing test. Defaults to False.

        Returns:
            pd.DataFrame: Results for each neuron
        """
        avg_trace = self.transformer.average_trace_from_aligned_binned(
            df_binned_aligned
        )
        if z:
            avg_trace = avg_trace.apply(zscore)
        df_res = avg_trace.apply(self._comp_one_col, window=self.window).round(3)
        self.responders = df_res.transpose().assign(p=lambda x: p_adjust(x["p"]))
        return self.responders

    def plot_responders(
        self,
        responders: pd.DataFrame = None,
        clusters: pd.DataFrame = None,
        bins: Any = "auto",
        cmap: Any = None,
    ) -> Union[plt.Axes, sns.FacetGrid]:
        """Plot Responders

        Args:
            responders (pd.DataFrame, optional): Output of get_responders method. Defaults to None.
            clusters (pd.DataFrame, optional): [neuron_id, wf_3]. Defaults to None.
            bins (Any, optional): To be passed to sns.histplot. Defaults to "auto".

        Returns:
            Union[plt.Axes, sns.FacetGrid]: Plot Object. Facet Grid if clusters specified.
        """
        if responders is None:
            responders = self.responders
        dfp = responders.reset_index().copy()
        dfp["Sig"] = dfp["p"] < 0.05
        if cmap is None:
            cmap = PAL_GREY_BLACK

        if clusters is None:
            _, ax = plt.subplots(figsize=(5, 5))
            ax = sns.histplot(
                data=dfp,
                x="Diff",
                multiple="stack",
                alpha=1,
                bins=bins,
                ax=ax,
                hue="Sig",
                hue_order=[True, False],
                palette=cmap,
                # edgecolor="black",
            )
            ax.set_xlabel("Spike Rate Change\n[Z Score]")
            ax.set_ylabel("Unit Counts")
            return ax
        else:
            dfp = dfp.merge(clusters)
            g = sns.FacetGrid(dfp, row="wf_3", sharey=False, aspect=2).map_dataframe(
                sns.histplot,
                x="Diff",
                hue="Sig",
                multiple="stack",
                alpha=1,
                bins=bins,
                hue_order=[True, False],
                palette=cmap,
                # edgecolor="black",
            )
            g.set_axis_labels(x_var="Spike Rate Change\n[Z Score]", y_var="Unit Counts")
            return g

    @staticmethod
    def _comp_one_col(ser, window, names=("Pre", "Post")):
        x = ser[(ser.index < window[0]) | (ser.index > window[1])]
        y = ser[~((ser.index < window[0]) | (ser.index > window[1]))]
        return mannwhitneyu_plusplus(x, y, names=names)


class ShortTsWilcox(ShortTSResponders):
    def get_responders(self, df_binned_aligned):
        # summary func pre post by trial. adjusted AUC by default.
        ...


class EvokedCounts:
    def __init__(self, window: Tuple[float, float] = (0.05, 0.3)):
        self.window = window

    def get_evoked_counts(self, df_binned_piv: pd.DataFrame):
        df_binned_piv = df_binned_piv.loc[lambda x: x["bin"].between(0.05, 0.2)]
        df_binned_piv = df_binned_piv.drop("bin", axis=1)
        df_long = df_binned_piv.melt(id_vars="event", var_name="neuron_id")
        evoked_counts = df_long.groupby(["neuron_id", "event"], as_index=False).sum()
        return evoked_counts

    def cov_evoked_counts(
        self, evoked_counts: pd.DataFrame, responders: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        cov = (
            evoked_counts.groupby("neuron_id")["value"]
            .apply(variation)
            .to_frame("cov")
            .reset_index()
        )
        if responders is not None:
            cov = cov.merge(responders.reset_index()).query("p < 0.05")
        return cov

    def plot_neuron(
        self,
        evoked_counts: pd.DataFrame,
        neuron: int,
        ax: Optional[plt.Axes] = None,
        bins="auto",
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax = evoked_counts.query("neuron_id == @neuron")["value"].hist(
            color="black", bins=bins
        )
        ax.set_ylabel("Trial Counts")
        ax.set_xlabel("Evoked Spikes")
        sns.despine()
        return ax


class ShockPlotter:
    def __init__(self):
        self.transformer = ShockUtils()

    def psth_heatmap_all(
        self,
        df_binned_piv: pd.DataFrame,
        responders: Optional[pd.DataFrame] = None,
        clusters: Optional[pd.DataFrame] = None,
        figsize=(5, 5),
    ) -> plt.Axes:
        # create psth
        _, ax = plt.subplots(figsize=figsize)
        df_avg = self.transformer.average_trace_from_aligned_binned(df_binned_piv)
        df_avg = df_avg.apply(gaussian_filter1d, sigma=1.5)
        df_avg = df_avg.apply(zscore)
        df_avg = df_avg.transpose()
        # order psth by responders
        if responders is not None:

            responders = responders.reset_index().loc[
                lambda x: x.neuron_id.isin(df_avg.index.values)
            ]
            if clusters is None:
                idx = responders.sort_values("Diff").neuron_id.values
            else:
                idx = (
                    responders.merge(clusters)
                    .sort_values(["wf_3", "Diff"])
                    .neuron_id.values
                )
            df_avg = df_avg.reindex(idx)
        ax = sns.heatmap(
            df_avg,
            center=0,
            cbar=False,
            xticklabels=[],
            yticklabels=[],
            robust=True,
            cmap="vlag",
            vmin=-2,
            vmax=2,
        )
        ax.axvline(50, color="black")
        ax.axis("off")
        return ax

    def psth_heatmap_by_cluster(
        self,
        df_binned_piv: pd.DataFrame,
        responders: Optional[pd.DataFrame] = None,
        clusters: Optional[pd.DataFrame] = None,
        figsize=(5, 5),
        title=True,
    ):
        from drn_interactions.neuron_transforms import ClusterUtils

        # df_avg = self.transformer.average_trace_from_aligned_binned(df_binned_piv)
        df_binned_piv = df_binned_piv.set_index(["event", "bin"])
        df_by_cluster = ClusterUtils().cluster_from_piv(df_binned_piv, clusters)
        figs = {}
        for cluster, df in df_by_cluster.items():
            ax = self.psth_heatmap_all(
                df.reset_index(), responders=responders, figsize=figsize
            )
            if title:
                ax.set_title(cluster)
            figs[cluster] = ax.get_figure()
        return figs

    def unit_raster_across_trials(
        self,
        df_spikes_aligned: pd.DataFrame,
        neuron: int,
        start_trial: int = 0,
        max_trial: Optional[int] = None,
    ):
        df_neuron = df_spikes_aligned.query("neuron_id == @neuron")
        trains = [g["aligned"].values for _, g in df_neuron.groupby("event")]
        trains = trains[start_trial:]
        if max_trial:
            trains = trains[: max_trial + 1]
        _, ax = plt.subplots(figsize=(5, 2), nrows=1, sharex=True)
        ax.eventplot(
            trains, color="black",
        )
        ax.axvline(0, color="red")
        ax.set_yticks([])
        ax.axis("off")
        ax.set_xticks([-0.5, 0, 1.5])
        return ax

    def unit_heatmap_across_trials(self):
        ...

    def population_single_trial(self):
        ...

    def population_heatmap_across_trials(self):
        ...
