from drn_interactions.stats import mannwhitneyu_plusplus, p_adjust
from drn_interactions.plots import PAL_GREY_BLACK
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import zscore
import pingouin as pg
from pymer4.models import Lmer
import seaborn as sns


class DrugResponders:
    def get_anova(
        self,
        df_binned: pd.DataFrame,
        clusters: Optional[pd.DataFrame] = None,
        z: bool = True,
    ):
        if z:
            df_binned["counts"] = df_binned.groupby("neuron_id")["counts"].apply(zscore)
        if clusters is None:
            anova = pg.mixed_anova(
                data=df_binned,
                dv="counts",
                within="block",
                between="drug",
                subject="neuron_id",
            ).round(3)
            return anova
        else:
            df_binned = df_binned.merge(clusters)
            mod = Lmer(
                "counts ~ block + drug + wf_3 + block:drug + wf_3:block + wf_3:drug + wf_3:block:drug + (drug | neuron_id) + (block | neuron_id)",
                data=df_binned,
            )
            mod.fit(
                factors={
                    "wf_3": ["sr", "sir", "ff"],
                    "block": ["pre", "post"],
                    "drug": ["sal", "cit"],
                },
                summarize=False,
            )
            anova = mod.anova()
            anova["effect"] = [
                "block",
                "drug",
                "neurontype",
                "block * drug",
                "neurontype * block",
                "neurontype * drug",
                "neurontype * block * drug",
            ]
            anova = anova[["effect", "NumDF", "DenomDF", "F-stat", "P-val", "Sig"]]
            coefs = (
                mod.coefs.reset_index()
                .assign(index=lambda x: x["index"].str.replace("block1", "post"))
                .assign(index=lambda x: x["index"].str.replace("drug1", "cit"))
                .assign(index=lambda x: x["index"].str.replace("wf_31", "sir"))
                .assign(index=lambda x: x["index"].str.replace("wf_32", "ff"))
                .assign(index=lambda x: x["index"].str.replace(":", " * "))
            )
            return anova, coefs

    def get_responders(
        self,
        df_binned: pd.DataFrame,
        clusters: Optional[pd.DataFrame] = None,
        z: bool = False,
        abs_diff_thresh: float = 0,
    ):
        # if clusters is not None:
        if z:
            df_binned["counts"] = df_binned.groupby("neuron_id")["counts"].transform(
                zscore
            )
        responders = (
            df_binned.groupby(["neuron_id", "drug"])
            .apply(self._mwu_prepost)
            .assign(p=lambda x: p_adjust(x["p"]))
            .assign(sig=lambda x: x["p"] < 0.05)
            .reset_index()
            .round(3)
        )
        responders["sig"] = np.where(
            responders["Diff"].abs() < abs_diff_thresh, False, responders["sig"]
        )
        if clusters is not None:
            responders = responders.merge(clusters)
        return responders

    @staticmethod
    def _mwu_prepost(df):
        x = df.query("block == 'pre'")["counts"].values
        y = df.query("block == 'post'")["counts"].values
        return mannwhitneyu_plusplus(x, y, names=("pre", "post"))

    def plot_responders(
        self,
        responders: pd.DataFrame,
        clusters: Optional[pd.DataFrame] = None,
        bins="auto",
        aspect=2,
        height=3,
    ) -> sns.FacetGrid:
        if clusters is None:
            g = sns.FacetGrid(
                responders,
                row="wf_3",
                sharey=False,
                sharex=True,
                aspect=aspect,
                height=height,
            )
        else:
            responders = responders.merge(clusters)
            g = sns.FacetGrid(
                responders,
                row="wf_3",
                col="drug",
                sharey=False,
                sharex=True,
                aspect=aspect,
                height=height,
            )
        return g.map_dataframe(
            sns.histplot,
            x="Diff",
            hue="sig",
            multiple="stack",
            alpha=1,
            bins=bins,
            hue_order=[True, False],
            palette=PAL_GREY_BLACK,
        )

