from typing import Tuple, Union, Dict, Optional, List
from .spikes import SpikesHandlerMulti
from .stats import p_adjust
from .plots import PAL_GREY_BLACK
from .load import load_neurons_derived, get_shock_sessions
from pymer4.models import Lmer
from rpy2.robjects import pandas2ri
import numpy as np
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from tqdm import tqdm
from matplotlib.figure import Figure


def load_slow_ts_data(
    session_names: Optional[List[str]] = None, bin_width: float = 1,
):
    neurons = load_neurons_derived().merge(get_shock_sessions())
    spikes = SpikesHandlerMulti(
        block=["base_shock", "post_base_shock"],
        bin_width=bin_width,
        session_names=session_names,
        t_start=-600,
        t_stop=1200,
    ).binned_piv
    spikes = spikes[
        [c for c in spikes.columns if c in neurons["neuron_id"].unique().tolist()]
    ]
    neurons = neurons.query("neuron_id in @spikes.columns")
    clusters = neurons[["neuron_id", "wf_3", "session_name"]]
    block = np.where(
        spikes.index < 0, "pre", np.where(spikes.index < 600, "shock", "post")
    )
    block = pd.Series(block, index=spikes.index, name="block")
    return spikes, block, clusters


class SlowTSResponders:
    def __init__(self, remove_empty=True):
        self.remove_empty = remove_empty

    @staticmethod
    def remove_empty_piv(df_piv):
        # move to a spikes class
        sums = df_piv.sum(axis=1)
        idx = sums[sums == 0].index.values
        return df_piv.copy().loc[lambda x: ~x.index.isin(idx)]

    @staticmethod
    def remove_empty_long(df_long, value_col):
        # move to a spikes class
        return df_long.copy().loc[lambda x: x[value_col] != 0]

    def get_responders(
        self,
        df_binned_piv: pd.DataFrame,
        clusters: pd.DataFrame = None,
        exclude_transition: bool = False,
    ):
        ...


class SlowTSRespondersMixed(SlowTSResponders):
    def get_responders(
        self,
        df_binned_piv: pd.DataFrame,
        clusters: pd.DataFrame = None,
        exclude_transition=False,
    ):
        pandas2ri.activate()
        out = {}

        if self.remove_empty:
            df_binned_piv = self.remove_empty_piv(df_binned_piv)

        df_binned_piv = df_binned_piv.apply(zscore)
        df1 = df_binned_piv.reset_index().melt(id_vars="bin", var_name="neuron_id")

        df1["block"] = np.where(
            df1["bin"] < 0, "pre", np.where(df1["bin"] < 600, "1", "2")
        )
        if exclude_transition:
            df1 = df1.loc[~df1.bin.between(550, 800)]

        if clusters is None:
            self.formula = "value ~ block + (block | neuron_id)"
            self.mod = Lmer(self.formula, data=df1)
            self.mod.fit(factors={"block": ["pre", "1", "2"]}, summarize=False)
            out["emms"], out["contrasts"] = self.mod.post_hoc(marginal_vars="block")
        else:
            df1 = df1.merge(clusters)
            self.formula = "value ~ block * wf_3 + (block | neuron_id)"
            self.mod = Lmer(self.formula, data=df1)
            self.mod.fit(
                factors={"block": ["pre", "1", "2"], "wf_3": ["sr", "sir", "ff"]},
                summarize=False,
            )
            out["emms"], out["contrasts"] = self.mod.post_hoc(
                marginal_vars="block", grouping_vars="wf_3"
            )

        out["coefs"] = self.mod.coefs.round(3)
        out["anova"] = self.mod.anova().round(3)
        out["slopes"] = self.mod.fixef
        out["slope_deviations"] = self.mod.ranef
        self.results_ = out
        return self

    def plot(self):
        return self.mod.plot_summary()


class SlowTSRespondersAnova(SlowTSResponders):
    def get_responders(
        self,
        df_binned_piv: pd.DataFrame,
        clusters: pd.DataFrame = None,
        exclude_transition=False,
        fit_unit_level_models=False,
    ):
        if self.remove_empty:
            df_binned_piv = self.remove_empty_piv(df_binned_piv)

        df_binned_piv = df_binned_piv.apply(zscore)
        df1 = df_binned_piv.reset_index().melt(id_vars="bin", var_name="neuron_id")

        df1["block"] = np.where(
            df1["bin"] < 0, "1pre", np.where(df1["bin"] < 600, "2shock", "3post")
        )
        if exclude_transition:
            df1 = df1.loc[~df1.bin.between(550, 800)]

        out = {}
        # repeated measures and post hoc tests
        if clusters is None:
            out["anova"] = pg.rm_anova(
                data=df1, dv="value", within="block", subject="neuron_id"
            ).round(3)
            out["contrasts"] = (
                pg.pairwise_ttests(
                    data=df1,
                    dv="value",
                    within="block",
                    subject="neuron_id",
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
            df1 = df1.merge(clusters)
            out["anova"] = pg.mixed_anova(
                data=df1,
                dv="value",
                within="block",
                between="wf_3",
                subject="neuron_id",
            ).round(3)
            out["contrasts"] = (
                pg.pairwise_ttests(
                    data=df1,
                    dv="value",
                    within="block",
                    between="wf_3",
                    subject="neuron_id",
                    interaction=True,
                    within_first=True,
                )
                .query("Contrast != 'wf_3' and block != '1pre'")
                .assign(p=lambda x: p_adjust(x["p-unc"]))
                .assign(Sig=lambda x: np.where(x.p < 0.05, "*", ""))
                .drop(
                    ["Paired", "Parametric", "alternative", "p-unc", "BF10", "hedges"],
                    axis=1,
                )
                .round(3)
            )

        if fit_unit_level_models:
            pairwise = []
            anovas = []
            for neuron in tqdm(df1.neuron_id.unique()):
                dfsub = df1.query("neuron_id == @neuron")
                df_anova = pg.anova(data=dfsub, dv="value", between="block").round(3)
                anovas.append(df_anova.assign(neuron_id=neuron))
                df_pairwise = pg.pairwise_tukey(
                    data=dfsub, dv="value", between="block",
                ).round(3)
                pairwise.append(df_pairwise.assign(neuron_id=neuron))

            out["unit_anovas"] = pd.concat(anovas).reset_index(drop=True)
            out["unit_anovas"] = out["unit_anovas"].assign(
                p_adj=lambda x: pg.multicomp(x["p-unc"].values, method="fdr_bh")[1]
            )
            sig_neurons = out["unit_anovas"].query("p_adj < 0.05").neuron_id.unique()

            unit_contrasts = pd.concat(pairwise).reset_index(drop=True)
            unit_contrasts = unit_contrasts.loc[lambda x: x.neuron_id.isin(sig_neurons)]
            unit_contrasts["diff_inv"] = unit_contrasts["diff"] * -1
            out["pre_to_shock"] = (
                unit_contrasts.query("A == '1pre' and B == '2shock'")
                .copy()
                .assign(
                    p_adj=lambda x: pg.multicomp(x["p-tukey"].values, method="fdr_bh")[
                        1
                    ]
                )
            )
            out["shock_to_post"] = (
                unit_contrasts.query("A == '2shock' and B == '3post'")
                .copy()
                .assign(
                    p_adj=lambda x: pg.multicomp(x["p-tukey"].values, method="fdr_bh")[
                        1
                    ]
                )
            )
            out["pre_to_post"] = (
                unit_contrasts.query("A == '1pre' and B == '3post'")
                .copy()
                .assign(
                    p_adj=lambda x: pg.multicomp(x["p-tukey"].values, method="fdr_bh")[
                        1
                    ]
                )
            )

        self.results_ = out

        return self

    def _plot_single_unit_effects(
        self,
        pre_to_shock: pd.DataFrame,
        shock_to_post: pd.DataFrame,
        pre_to_post: pd.DataFrame,
    ) -> Figure:
        f, ax = plt.subplots(nrows=3, figsize=(5, 8), sharey=False, sharex=True)
        sns.histplot(
            data=pre_to_shock.assign(sig=lambda x: x["p_adj"] < 0.05),
            x="diff_inv",
            hue="sig",
            ax=ax[0],
            alpha=1,
            bins=50,
            palette=PAL_GREY_BLACK[::-1],
        )

        sns.histplot(
            data=shock_to_post.assign(sig=lambda x: x["p_adj"] < 0.05),
            x="diff_inv",
            hue="sig",
            ax=ax[1],
            alpha=1,
            bins=50,
            palette=PAL_GREY_BLACK[::-1],
        )

        sns.histplot(
            data=pre_to_post,
            x="diff_inv",
            hue="sig",
            ax=ax[2],
            alpha=1,
            bins=50,
            palette=PAL_GREY_BLACK[::-1],
        )
        ax[0].legend().remove()
        ax[1].legend().remove()
        ax[2].legend().remove()

        ax[0].set_title("Pre to Shock")
        ax[1].set_title("Shock to Post")
        ax[2].set_title("Pre to Post")

        ax[2].set_xlabel("Spike Rate Change\n[Z Score]")
        sns.despine()
        plt.tight_layout()
        return f

    def plot_unit_effects(self, clusters=None,) -> Union[Figure, Dict[str, Figure]]:
        pre_to_shock = (
            self.results_["pre_to_shock"].copy().assign(sig=lambda x: x.p_adj < 0.05)
        )
        shock_to_post = (
            self.results_["shock_to_post"].copy().assign(sig=lambda x: x.p_adj < 0.05)
        )
        pre_to_post = (
            self.results_["pre_to_post"].copy().assign(sig=lambda x: x.p_adj < 0.05)
        )
        if clusters is not None:
            out: Dict[str, Figure] = {}
            pre_to_shock = pre_to_shock.merge(clusters)
            shock_to_post = shock_to_post.merge(clusters)
            pre_to_post = pre_to_post.merge(clusters)
            for cluster in clusters["wf_3"].unique():
                out[cluster] = self._plot_single_unit_effects(
                    pre_to_shock.query("wf_3 == @cluster"),
                    shock_to_post.query("wf_3 == @cluster"),
                    pre_to_post.query("wf_3 == @cluster"),
                )
            return out
        else:
            return self._plot_single_unit_effects(
                pre_to_shock, shock_to_post, pre_to_post
            )


class SlowTSRespondersMWU(SlowTSResponders):
    ...


def plot_pop(
    session_name: str,
    bin_width: float = 10,
    figsize: Tuple[float, float] = (5, 3),
    title: bool = True,
    t_start: float = -600,
    t_stop: float = 1200,
    z: bool = False,
) -> plt.Axes:
    _, ax = plt.subplots(figsize=figsize)
    df = SpikesHandlerMulti(
        block=["base_shock", "post_base_shock"],
        bin_width=bin_width,
        session_names=[session_name],
        t_start=t_start,
        t_stop=t_stop,
    ).binned_piv
    pop = df.sum(axis=1).to_frame()
    if z:
        pop = pop.apply(zscore)
    ax = pop.plot(ax=ax, color="black")
    if title:
        ax.set_title(session_name)
    sns.despine()
    return ax


def population_raster(
    session_name: str,
    clusters: Optional[pd.DataFrame] = None,
    bin_width: float = 1,
    figsize: Tuple[float, float] = (5, 3),
    title: bool = True,
    t_start: float = -600,
    t_stop: float = 1200,
    tfidf: bool = False,
) -> plt.Axes:
    _, ax = plt.subplots(figsize=figsize)
    df = SpikesHandlerMulti(
        block=["base_shock", "post_base_shock"],
        session_names=[session_name],
        bin_width=bin_width,
        t_start=t_start,
        t_stop=t_stop,
    ).binned_piv
    if clusters is not None:
        session_neurons = df.columns
        clusters = clusters.loc[lambda x: x.neuron_id.isin(session_neurons)]
        clusters = clusters.sort_values("wf_3")
        idx = clusters.neuron_id.values.tolist()
        df = df[idx]
    vals = df.values.T
    if tfidf:
        vals = TfidfTransformer().fit_transform(vals.T).toarray().T
    sns.heatmap(
        data=vals,
        cmap="Greys",
        ax=ax,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        robust=True,
    )
    ax.axis("off")
    if title:
        ax.set_title(session_name)
    return ax
