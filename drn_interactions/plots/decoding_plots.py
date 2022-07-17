import numpy as np
import seaborn as sns
from drn_interactions.io import load_neurons_derived
import matplotlib.pyplot as plt
import pingouin as pg


def pop_unit_hist(df_pop, df_units, bins=None, ax_pop=None, ax_unit=None):
    if ax_pop is None or ax_unit is None:
        f, (ax_pop, ax_unit) = plt.subplots(
            2, 1, figsize=(6, 5), sharex=True, constrained_layout=True
        )
    if bins is None:
        bins = np.arange(0.3, 1.1, 0.05)

    ax_unit.hist(df_units["F1 Score"], color="black", bins=bins, density=False)
    ax_unit.set_ylabel("Unit\nCount")

    ax_pop.hist(df_pop["true"], color="black", bins=bins, density=False)
    ax_pop.set_xlabel("Decoder Perforance\n(F1 Score)")
    ax_pop.set_ylabel("Session\nCount")

    sns.despine(ax=ax_pop)
    sns.despine(ax=ax_unit)
    return ax_unit, ax_pop


def compare_best_unit(df_unit, df_pop_long, ax=None, yticks=None, ylim=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # get best unit per session
    neurons = load_neurons_derived()[["neuron_id", "session_name"]]
    df_best_unit = (
        df_unit.merge(neurons)
        .groupby("session_name")["F1 Score"]
        .max()
        .to_frame("Best Unit")
        .reset_index()
    )

    # rename pop to have population as f1 score
    df_pop_long = df_pop_long.copy().rename(columns={"F1 Score": "Population"})

    # create merged df
    df = df_best_unit.merge(df_pop_long)
    df = df.melt(
        id_vars=["session_name", "shuffle_status"],
        var_name="Decoder Type",
        value_name="F1 Score",
    )

    # plot
    pg.plot_paired(
        data=df,
        dv="F1 Score",
        within="Decoder Type",
        subject="session_name",
        ax=ax,
        boxplot=True,
        **kwargs
    )
    ax.set_ylabel("Decoder Performance\n(F1 Score)")
    ax.set_xlabel(None)
    return ax


def plot_performance_vs_n_units(df_pop_n, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    sns.pointplot(
        data=df_pop_n.reset_index(drop=True),
        x="n_units",
        y="true",
        color="black",
        ax=ax,
        scale=0.5,
        errwidth=2.5,
        **kwargs
    )
    ax.set_ylim(0.45, 1)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xlabel("Number of Units")
    ax.set_ylabel("F1 Score")
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    sns.despine(ax=ax)
    return ax


def dropout_boxplots(df_pop_nt, ax=None):
    PROPS = {
        "boxprops": {"facecolor": "grey", "edgecolor": "black"},
        "medianprops": {"color": "black"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
    }
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    ax = sns.boxplot(
        data=df_pop_nt.reset_index(drop=True),
        x="dropout",
        y="true",
        ax=ax,
        color="white",
        width=0.3,
        fliersize=1,
        linewidth=3,
        **PROPS
    )
    ax.set_ylim(0.45, 1)
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Neuron Type Dropped Out")
    ax.set_xticklabels(["SR", "SIR", "FF"])
    sns.despine(ax=ax)
    return ax
