from pymer4.models import Lmer
from pymer4.utils import con2R
from rpy2.robjects import pandas2ri
import numpy as np
from scipy.stats import zscore
import pandas as pd
import pingouin as pg
from tqdm import tqdm


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

    def get_responders(**kwargs):
        ...


class SlowTSRespondersMixed(SlowTSResponders):
    def get_responders(
        self, df_binned_piv: pd.DataFrame, clusters=None, exclude_trasition=False
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
        if exclude_trasition:
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
        clusters=None,
        fit_unit_level_models=False,
        exclude_trasition=False,
    ):
        if self.remove_empty:
            df_binned_piv = self.remove_empty_piv(df_binned_piv)

        df_binned_piv = df_binned_piv.apply(zscore)
        df1 = df_binned_piv.reset_index().melt(id_vars="bin", var_name="neuron_id")

        df1["block"] = np.where(
            df1["bin"] < 0, "1pre", np.where(df1["bin"] < 600, "2shock", "3post")
        )
        if exclude_trasition:
            df1 = df1.loc[~df1.bin.between(550, 800)]

        out = {}
        # repeated measures and post hoc tests
        if clusters is None:
            out["anova"] = pg.rm_anova(
                data=df1, dv="value", within="block", subject="neuron_id"
            ).round(3)
            out["contrasts"] = pg.pairwise_ttests(
                data=df1,
                dv="value",
                within="block",
                subject="neuron_id",
                padjust="fdr_bh",
            ).round(3)
        else:
            df1 = df1.merge(clusters)
            out["anova"] = pg.mixed_anova(
                data=df1,
                dv="value",
                within="block",
                between="wf_3",
                subject="neuron_id",
            ).round(3)
            out["contrasts"] = pg.pairwise_ttests(
                data=df1,
                dv="value",
                within="block",
                between="wf_3",
                subject="neuron_id",
                interaction=False,
            ).round(3)

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


class SlowTSRespondersMWU:
    ...
