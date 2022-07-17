import pingouin as pg
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency
from drn_interactions.stats import p_adjust, prop_of_total


class AnovaPostHoc:
    def __init__(self, neuron_type_col, value_col, round=None):
        self.neuron_type_col = neuron_type_col
        self.value_col = value_col
        self.round = round

    def anova(self, df):
        df_anova = pg.anova(data=df, dv=self.value_col, between=self.neuron_type_col)
        if self.round is not None:
            df_anova = df_anova.round(self.round)
        return df_anova

    def contrasts(self, df):
        df_contrasts = pg.pairwise_tukey(
            data=df, dv=self.value_col, between=self.neuron_type_col
        )
        if self.round is not None:
            df_contrasts = df_contrasts.round(self.round)
        return df_contrasts

    def extract(self, df_anova, df_contrasts):
        out = {}
        f = df_anova.iloc[0]["F"]
        f_dof = df_anova.iloc[0]["ddof1"]
        f_p = df_anova.iloc[0]["p-unc"]
        f_star = "*" if f_p < 0.05 else ""
        out["F"] = f"F({f_dof})={f} (p={f_p}{f_star})"

        for idx, row in df_contrasts.iterrows():
            out.update(self._extract_contrasts_row(row))
        return pd.Series(out)

    def _extract_contrasts_row(self, row):
        means = f"{row['mean(A)']}; {row['mean(B)']}"
        contrast = f"{row['A']} - {row['B']}"
        star = "*" if row["p-tukey"] < 0.05 else ""
        t = f"T={row['T']} (p={row['p-tukey']}{star})"
        return {contrast: f"{means} | {t}"}

    def __call__(self, df):
        df_anova = self.anova(df)
        df_contrasts = self.contrasts(df)
        return self.extract(df_anova, df_contrasts)


class ChiSquarePostHoc:
    def __init__(
        self,
        neuron_type_col="neuron_type",
        value_col="value",
        p_adjust_method="Benjamini-Hochberg",
        round=None,
    ):
        self.neuron_type_col = neuron_type_col
        self.value_col = value_col
        self.p_adjust_method = p_adjust_method
        self.round = round

    def chi_square(self, df):
        df_counts = pd.crosstab(df[self.neuron_type_col], df[self.value_col])
        stat, p, dof, expected = chi2_contingency(df_counts)
        if self.round is not None:
            stat = round(stat, self.round)
            p = round(p, self.round)
        return stat, p, dof, expected

    def contrasts(self, df):
        combs = list(combinations(df[self.neuron_type_col].unique(), 2))
        p_vals = np.empty(len(combs))
        stats = np.empty(len(combs))
        contrasts = np.empty(len(combs), dtype=np.object)
        dofs = np.empty(len(combs))
        prop1s = np.empty(len(combs))
        prop2s = np.empty(len(combs))
        for i, (ntype1, ntype2) in enumerate(combs):
            df_sub = df.query(
                f"{self.neuron_type_col} == '{ntype1}' or {self.neuron_type_col} == '{ntype2}'"
            )
            df_counts_sub = pd.crosstab(
                df_sub[self.neuron_type_col], df_sub[self.value_col]
            ).reindex((ntype1, ntype2))
            stat, p, dof, _ = chi2_contingency(df_counts_sub)
            prop1, prop2 = df_counts_sub.apply(prop_of_total, axis=1).iloc[:, -1].values
            prop1s[i] = prop1 * 100
            prop2s[i] = prop2 * 100
            contrasts[i] = f"{ntype1} - {ntype2}"
            p_vals[i] = p
            stats[i] = stat
            dofs[i] = dof
        p_vals = p_adjust(np.array(p_vals))
        if self.round is not None:
            p_vals = p_vals.round(self.round)
            stats = stats.round(self.round)
            dofs = dofs.round(self.round)
            prop1s = prop1s.round(self.round)
            prop2s = prop2s.round(self.round)

        return p_vals, stats, dofs, contrasts, prop1s, prop2s

    def extract(
        self,
        dof_anova,
        stat_anova,
        p_anova,
        p_vals_contrats,
        stats_contrats,
        labels_contrasts,
        dof_contrasts,
        prop1s_contrasts,
        prop2s_contrasts,
    ):
        out = {}
        star_anova = "*" if p_anova < 0.05 else ""
        out[
            "anova"
        ] = f"Chi2({dof_anova})={stat_anova:.1f} (p={p_anova:.2f}{star_anova})"

        stars_contrasts = ["*" if p < 0.05 else "" for p in p_vals_contrats]
        dict_contrasts = {
            contrast: f"{p1}%; {p2}% | Chi({dof})={stat} (p={pval}{star})"
            for contrast, dof, stat, pval, star, p1, p2 in zip(
                labels_contrasts,
                dof_contrasts,
                stats_contrats,
                p_vals_contrats,
                stars_contrasts,
                prop1s_contrasts,
                prop2s_contrasts,
            )
        }
        out.update(dict_contrasts)
        return pd.Series(out)

    def __call__(self, df):
        stat_anova, p_anova, dof_anova, _ = self.chi_square(df)
        (
            p_vals_contrats,
            stats_contrats,
            dof_contrasts,
            labels_contrasts,
            prop1s,
            prop2s,
        ) = self.contrasts(df)
        out = self.extract(
            dof_anova,
            stat_anova,
            p_anova,
            p_vals_contrats,
            stats_contrats,
            labels_contrasts,
            dof_contrasts,
            prop1s,
            prop2s,
        )
        return out
