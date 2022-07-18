from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Any, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from drn_interactions.plots import heatmap, wrap_all_text, wrap_ylabel, PAL_GREY_BLACK
from sklearn.preprocessing import robust_scale
from drn_interactions.config import Config
from drn_interactions.io import load_waveforms
from drn_interactions.transforms import SpikesHandler
from drn_interactions.spiketrains.waveforms import WaveformPreprocessor
import matplotlib as mpl
import textwrap


class NeuronTypesFigureLoader:
    def __init__(
        self,
        example_session_name: str,
        example_session_t_start: float,
        example_session_length: float = 10,
        probe_neuron_types_path: Optional[Path] = None,
        probe_waveform_props_path: Optional[Path] = None,
        probe_spiketrain_props_path: Optional[Path] = None,
        single_unit_props_path: Optional[Path] = None,
        single_unit_neuron_types_path: Optional[Path] = None,
    ):
        self.example_session_name = example_session_name
        self.example_session_t_start = example_session_t_start
        self.example_session_length = example_session_length
        self.probe_neuron_types_path = (
            probe_neuron_types_path
            if probe_neuron_types_path is not None
            else Config.derived_data_dir / "neuron_types.csv"
        )
        self.probe_waveform_props_path = (
            probe_waveform_props_path
            if probe_waveform_props_path is not None
            else Config.derived_data_dir / "waveform_summary.csv"
        )
        self.probe_spiketrain_props_path = (
            probe_spiketrain_props_path
            if probe_spiketrain_props_path is not None
            else Config.derived_data_dir / "spiketrain_stats_segments.csv"
        )
        self.single_unit_props_path = (
            single_unit_props_path
            if single_unit_props_path is not None
            else Config.derived_data_dir / "single_unit_dataset_tidied.csv"
        )
        self.single_unit_neuron_types_path = (
            single_unit_neuron_types_path
            if single_unit_neuron_types_path is not None
            else Config.derived_data_dir / "neuron_types_single_unit.csv"
        )

    def load_neuron_types(self) -> pd.DataFrame:
        return pd.read_csv(self.probe_neuron_types_path)

    def load_waveforms(self) -> pd.DataFrame:
        return load_waveforms()

    def load_waveform_props(self) -> pd.DataFrame:
        return pd.read_csv(self.probe_waveform_props_path)

    def load_spiketrain_props(self) -> pd.DataFrame:
        return pd.read_csv(self.probe_spiketrain_props_path)

    def load_single_unit_props(self) -> pd.DataFrame:
        return pd.read_csv(self.single_unit_props_path)

    def load_single_unit_neuron_types(self) -> pd.DataFrame:
        return pd.read_csv(self.single_unit_neuron_types_path)

    def load_spikes(self) -> pd.DataFrame:
        sh = SpikesHandler(
            block="pre",
            session_names=[self.example_session_name],
            bin_width=10,
            t_start=self.example_session_t_start,
            t_stop=self.example_session_t_start + self.example_session_length,
        )
        return sh.spikes

    def load_all(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        probe_neuron_types = self.load_neuron_types()
        probe_waveforms = self.load_waveforms()
        probe_waveform_props = self.load_waveform_props()
        probe_spiketrain_props = self.load_spiketrain_props()
        single_unit_props = self.load_single_unit_props()
        single_unit_neuron_types = self.load_single_unit_neuron_types()
        spikes = self.load_spikes()
        return (
            probe_neuron_types,
            probe_waveforms,
            probe_waveform_props,
            probe_spiketrain_props,
            single_unit_props,
            single_unit_neuron_types,
            spikes,
        )


class NeuronTypesFigurePreprocessor:
    def __init__(
        self,
        electrode_type_order: Sequence[str] = ("Silicon Probe", "Glass Pipette"),
        neuron_type_order: Sequence[str] = ("SR", "SIR", "FF"),
        spikes_id_col: str = "neuron_id",
        spikes_spiketime_col: str = "spiketimes",
        min_spikes_per_train: int = 2,
        max_spikes_per_train: int = 150,
        new_col_names: Sequence[str] = (
            "ID",
            "Neuron Type",
            "Waveform Width (ms)",
            "CV(ISI)",
            "Spike Rate (Hz)",
        ),
        metrics: Sequence[str] = (
            "Waveform Width (ms)",
            "CV(ISI)",
            "Spike Rate (Hz)",
        ),
        current_probe_cols: Sequence[str] = (
            "neuron_id",
            "neuron_type",
            "width_basepost",
            "cv_isi_burst",
            "mean_firing_rate",
        ),
        current_single_unit_cols: Sequence[str] = (
            "id",
            "neuron_type",
            "to_baseline",
            "cv-isi",
            "firing_rate",
        ),
        created_electrode_type_col: str = "Electrode Type",
        created_metric_col_name: str = "Metric",
        created_value_col_name: str = "Value",
        probe_waveform_divisor: float = 30,
        probe_waveform_time_col: str = "waveform_index",
        created_probe_lab: str = "Silicon Probe",
        created_single_unit_lab: str = "Glass Pipette",
        id_col: str = "ID",
        neuron_type_col: str = "Neuron Type",
        waveform_preprocessor: Optional[WaveformPreprocessor] = None,
    ):
        self.min_spikes_per_train = min_spikes_per_train
        self.max_spikes_per_train = max_spikes_per_train
        self.spikes_id_col = spikes_id_col
        self.spikes_spiketime_col = spikes_spiketime_col
        self.electrode_type_order = electrode_type_order
        self.neuron_type_order = neuron_type_order
        self.new_col_names = new_col_names
        self.current_probe_cols = current_probe_cols
        self.current_single_unit_cols = current_single_unit_cols
        self.created_electrode_type_col = created_electrode_type_col
        self.created_metric_col_name = created_metric_col_name
        self.created_value_col_name = created_value_col_name
        self.probe_mapper = {
            probe_col: new_col
            for new_col, probe_col in zip(new_col_names, current_probe_cols)
        }
        self.single_unit_mapper = {
            single_unit_col: new_col
            for new_col, single_unit_col in zip(new_col_names, current_single_unit_cols)
        }
        self.probe_waveform_divisor = probe_waveform_divisor
        self.probe_waveform_time_col = probe_waveform_time_col
        self.created_probe_lab = created_probe_lab
        self.created_single_unit_lab = created_single_unit_lab
        self.meta_cols = [c for c in new_col_names if c not in metrics] + [
            created_electrode_type_col
        ]
        self.metrics = metrics
        self.id_col = id_col
        self.neuron_type_col = neuron_type_col
        self.waveform_preprocessor = (
            waveform_preprocessor
            if waveform_preprocessor is not None
            else WaveformPreprocessor()
        )

    def preprocess_probe(
        self,
        probe_spiketrain_props: pd.DataFrame,
        probe_waveform_props: pd.DataFrame,
        probe_neuron_types: pd.DataFrame,
    ) -> pd.DataFrame:
        df_probe = (
            probe_spiketrain_props.merge(probe_waveform_props, how="outer")
            .merge(probe_neuron_types)
            .assign(
                width_basepost=lambda x: x.width_basepost.divide(
                    self.probe_waveform_divisor
                )
            )
            .rename(columns=self.probe_mapper)
            .assign(**{self.created_electrode_type_col: self.created_probe_lab})
            .melt(
                id_vars=self.meta_cols,
                value_vars=self.metrics,
                var_name=self.created_metric_col_name,
                value_name=self.created_value_col_name,
            )
        )
        return df_probe

    def preprocess_single_unit(
        self,
        single_unit_props: pd.DataFrame,
        single_unit_neuron_types: pd.DataFrame,
    ) -> pd.DataFrame:
        df_single_unit = (
            single_unit_props.merge(single_unit_neuron_types)
            .rename(columns=self.single_unit_mapper)
            .assign(**{self.created_electrode_type_col: self.created_single_unit_lab})
            .melt(
                id_vars=self.meta_cols,
                value_vars=self.metrics,
                var_name=self.created_metric_col_name,
                value_name=self.created_value_col_name,
            )
            .assign(**{self.id_col: lambda x: x[self.id_col] + 100000})
        )
        return df_single_unit

    def make_categorical_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.created_electrode_type_col] = pd.Categorical(
            df[self.created_electrode_type_col], self.electrode_type_order, ordered=True
        )
        df[self.neuron_type_col] = pd.Categorical(
            df[self.neuron_type_col], self.neuron_type_order, ordered=True
        )
        return df

    def preprocess_waveforms(
        self, probe_waveforms: pd.DataFrame, df_probe_preprocessed: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        waveforms_pp = self.waveform_preprocessor(probe_waveforms).transpose()
        # waveforms_pp.columns = waveforms_pp.columns.map(str)

        out = {}
        for neuron_type in self.neuron_type_order:
            neurons = (
                df_probe_preprocessed.loc[
                    lambda x: x[self.neuron_type_col] == neuron_type
                ][self.id_col]
                .unique()
                .tolist()
            )
            out[neuron_type] = waveforms_pp.loc[:, neurons]
        return out

    def pivot_props(self, df_props: pd.DataFrame) -> pd.DataFrame:
        df_props = df_props.pivot_table(
            index=[self.created_electrode_type_col, self.neuron_type_col, self.id_col],
            columns=self.created_metric_col_name,
            values=self.created_value_col_name,
        )
        df_props = df_props.sort_index(level=1, ascending=False)
        return df_props

    def preprocess_spikes(
        self, probe_spikes: pd.DataFrame, df_probe_preprocessed: pd.DataFrame
    ) -> Dict[str, List[np.ndarray]]:

        out = {}
        for neuron_type in self.neuron_type_order:
            neurons = (
                df_probe_preprocessed.loc[
                    lambda x: x[self.neuron_type_col] == neuron_type
                ][self.id_col]
                .unique()
                .tolist()
            )
            probe_spikes_sub = probe_spikes.loc[
                lambda x: x[self.spikes_id_col].isin(neurons)
            ]
            trains = [
                g[self.spikes_spiketime_col].values
                for _, g in probe_spikes_sub.groupby(self.spikes_id_col)
                if len(g[self.spikes_spiketime_col]) >= self.min_spikes_per_train
                and len(g[self.spikes_spiketime_col]) <= self.max_spikes_per_train
            ]
            out[neuron_type] = trains
        return out

    def preprocess_all(
        self,
        probe_neuron_types: pd.DataFrame,
        probe_waveforms: pd.DataFrame,
        probe_waveform_props: pd.DataFrame,
        probe_spiketrain_props: pd.DataFrame,
        single_unit_props: pd.DataFrame,
        single_unit_neuron_types: pd.DataFrame,
        spikes: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[np.ndarray]]]:
        df_probe = self.preprocess_probe(
            probe_spiketrain_props,
            probe_waveform_props,
            probe_neuron_types,
        )
        df_single_unit = self.preprocess_single_unit(
            single_unit_props,
            single_unit_neuron_types,
        )
        df = pd.concat([df_probe, df_single_unit]).reset_index(drop=True)
        df_waveforms = self.preprocess_waveforms(probe_waveforms, df_probe)
        trains = self.preprocess_spikes(spikes, df_probe)
        return df, df_waveforms, trains


class NeuronPropsPlotter:
    def __init__(
        self,
        neuron_type_col: str = "Neuron Type",
        electride_type_col: str = "Electrode Type",
        neuron_id_col: str = "neuron_id",
        metric_col: str = "Metric",
        value_col: str = "Value",
        tick_fontsize: str = "medium",
    ):
        self.neuron_type_col = neuron_type_col
        self.neuron_id_col = neuron_id_col
        self.metric_col = metric_col
        self.value_col = value_col
        self.electride_type_col = electride_type_col
        self.tick_fontsize = tick_fontsize

    def neuron_type_heat(
        self,
        df,
        ax,
        cbar_ax=None,
        plot_cbar=True,
        formater_x=None,
        cmap="viridis",
        vmin=None,
        vmax=None,
        dropna=True,
        scaler=robust_scale,
    ):
        metrics = df[self.metric_col].value_counts().index.values.tolist()
        if formater_x is None:
            metrics[1] = "Rate (Hz)"
            metrics[-1] = "Width (ms)"
            formater_x = plt.FixedFormatter(metrics)

        locater_x = plt.IndexLocator(1, 0.5)
        df = (
            df.pivot(
                index=["Neuron Type", "neuron_id"], columns="Metric", values="Value"
            )
            .sort_index(level=0, ascending=True)
            .droplevel(0)
        )
        if dropna:
            df = df.dropna()
        df = df.apply(scaler)
        heatmap(
            df_binned_piv=df,
            heatmap_kwargs=dict(
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                cbar_ax=cbar_ax,
                cbar=plot_cbar,
            ),
            ax=ax,
            locater_x=locater_x,
            formater_x=formater_x,
            tick_params=dict(
                width=0, rotation=45, length=5, labelsize=self.tick_fontsize
            ),
            cbar_title_kwargs=dict(fontsize=self.tick_fontsize, label="Z-score"),
            cbar_tick_params=dict(labelsize=self.tick_fontsize, width=0, length=0),
        )
        return ax


class NeuronTypesFigureGridMaker:
    def make_figure(
        self,
        figsize: Tuple[float, float] = (7, 7),
        constrained_layout: bool = True,
        **kwargs
    ):
        f = plt.figure(figsize=figsize, constrained_layout=constrained_layout, **kwargs)
        return f

    def split_top_bottom(self, fig: mpl.figure.Figure, hspace: float = 0.3, **kwargs):
        fig_top, fig_bottom = fig.subfigures(nrows=2, hspace=hspace, **kwargs)
        return fig_top, fig_bottom

    def split_top_figs(
        self,
        fig: mpl.figure.Figure,
        wspace: float = 0.01,
        width_ratios=(1, 0.2),
        **kwargs
    ):
        fig_raster, fig_waveforms = fig.subfigures(
            ncols=2, wspace=wspace, width_ratios=width_ratios, **kwargs
        )
        return fig_raster, fig_waveforms

    def split_bottom_figs(
        self,
        fig: mpl.figure.Figure,
        wspace: float = 0.2,
        width_ratios=(0.5, 0.4, 0.2),
        **kwargs
    ):
        fig_bottom_left, fig_bottom_middle, fig_bottom_right = fig.subfigures(
            ncols=3, wspace=wspace, width_ratios=width_ratios, **kwargs
        )
        return fig_bottom_left, fig_bottom_middle, fig_bottom_right


class NeuronTypesPlotter:
    def __init__(self):
        ...

    def plot_raster(
        self,
        trains: Dict[str, List[np.ndarray]],
        fig: Optional[mpl.figure.Figure] = None,
        grid_spec_kwargs: Optional[Dict[str, Any]] = None,
        neuron_type_order: Optional[List[str]] = None,
        height_ratios: Optional[List[float]] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        turn_off_axis: bool = True,
    ):
        fig = fig or plt.figure()
        grid_spec_kwargs = grid_spec_kwargs or {}
        grid_spec_kwargs["hspace"] = grid_spec_kwargs.get("hspace", 0.1)
        grid_spec_kwargs["top"] = grid_spec_kwargs.get("top", 0.9)
        grid_spec_kwargs["bottom"] = grid_spec_kwargs.get("bottom", 0.1)
        grid_spec_kwargs["left"] = grid_spec_kwargs.get("left", 0.1)
        grid_spec_kwargs["right"] = grid_spec_kwargs.get("right", 0.9)

        plot_kwargs = plot_kwargs or {}
        plot_kwargs["color"] = plot_kwargs.get("color", "black")
        plot_kwargs["linewidths"] = plot_kwargs.get("linewidths", 1.5)
        plot_kwargs["linelengths"] = plot_kwargs.get("linelengths", 1)
        neuron_type_order = neuron_type_order or list(trains.keys())
        if height_ratios is None:
            height_ratios = [len(x) for x in trains.values()]
        gs = fig.add_gridspec(
            nrows=len(neuron_type_order),
            ncols=1,
            height_ratios=height_ratios,
            **grid_spec_kwargs,
        )
        axes = {}
        for neuron_type in neuron_type_order:
            train_nt = trains[neuron_type]
            axes[neuron_type] = fig.add_subplot(
                gs[neuron_type_order.index(neuron_type), 0]
            )
            axes[neuron_type].eventplot(train_nt, **plot_kwargs)
            if turn_off_axis:
                axes[neuron_type].axis("off")

    def plot_waveforms(
        self,
        waveforms: Dict[str, pd.DataFrame],
        fig: Optional[mpl.figure.Figure] = None,
        height_ratios: Optional[List[float]] = None,
        grid_spec_kwargs: Optional[Dict[str, Any]] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        turn_off_axis: bool = True,
        neuron_type_order: Optional[Sequence[str]] = None,
        waveform_idx: int = 0,
    ) -> Dict[str, mpl.axes.Axes]:
        fig = fig or plt.figure()
        grid_spec_kwargs = grid_spec_kwargs or {}
        grid_spec_kwargs["top"] = grid_spec_kwargs.get("top", 0.75)
        grid_spec_kwargs["bottom"] = grid_spec_kwargs.get("bottom", 0.1)
        grid_spec_kwargs["left"] = grid_spec_kwargs.get("left", 0)
        grid_spec_kwargs["right"] = grid_spec_kwargs.get("right", 0.6)
        grid_spec_kwargs["wspace"] = grid_spec_kwargs.get("wspace", 0)
        grid_spec_kwargs["hspace"] = grid_spec_kwargs.get("hspace", 0.8)

        plot_kwargs = plot_kwargs or {}
        plot_kwargs["color"] = plot_kwargs.get("color", "black")
        plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 1.5)
        neuron_type_order = neuron_type_order or list(waveforms.keys())
        if height_ratios is None:
            height_ratios = [1 for _ in range(len(neuron_type_order))]
        gs = fig.add_gridspec(
            nrows=len(neuron_type_order),
            ncols=1,
            height_ratios=height_ratios,
            **grid_spec_kwargs,
        )
        axes = {}
        for neuron_type in neuron_type_order:
            waveform = waveforms[neuron_type].iloc[:, waveform_idx]
            axes[neuron_type] = fig.add_subplot(
                gs[neuron_type_order.index(neuron_type), 0]
            )
            axes[neuron_type].plot(waveform.index, waveform.values, **plot_kwargs)
            if turn_off_axis:
                axes[neuron_type].axis("off")
        return axes

    def plot_heatmaps(
        self,
        df_props_probe: pd.DataFrame,
        df_props_single_unit: pd.DataFrame,
        fig: Optional[mpl.figure.Figure] = None,
        grid_spec_kwargs: Optional[Dict[str, Any]] = None,
        heatmap_kwargs: Optional[Dict[str, Any]] = None,
        tick_params: Optional[Dict[str, Any]] = None,
        cbar_kwargs: Optional[Dict[str, Any]] = None,
        textwrap_width: Optional[int] = None,
    ):
        # make axes grid
        fig = fig or plt.figure()
        grid_spec_kwargs = grid_spec_kwargs or {}
        grid_spec_kwargs["top"] = grid_spec_kwargs.get("top", 0.6)
        grid_spec_kwargs["bottom"] = grid_spec_kwargs.get("bottom", 0.3)
        grid_spec_kwargs["right"] = grid_spec_kwargs.get("right", 0.91)
        grid_spec_kwargs["left"] = grid_spec_kwargs.get("left", 0.05)
        grid_spec_kwargs["wspace"] = grid_spec_kwargs.get("wspace", 0.25)

        grid_spec_kwargs["width_ratios"] = grid_spec_kwargs.get(
            "width_ratios", [1, 1, 0.08]
        )
        grid_spec_kwargs["height_ratios"] = grid_spec_kwargs.get(
            "height_ratios", [0.5, 1]
        )
        axes = fig.subplot_mosaic(
            [
                [
                    "h1",
                    "h2",
                    "c",
                ],
                [
                    "h1",
                    "h2",
                    ".",
                ],
            ],
            gridspec_kw=grid_spec_kwargs,
        )

        # plot defaults
        heatmap_kwargs = heatmap_kwargs or {}
        heatmap_kwargs["cmap"] = heatmap_kwargs.get("cmap", "coolwarm")
        heatmap_kwargs["vmin"] = heatmap_kwargs.get("vmin", -1)
        heatmap_kwargs["vmax"] = heatmap_kwargs.get("vmax", 1)
        heatmap_kwargs["center"] = heatmap_kwargs.get("center", 0)
        heatmap_kwargs["xticklabels"] = heatmap_kwargs.get(
            "xticklabels", df_props_probe.columns
        )

        tick_params = tick_params or {}
        tick_params["width"] = heatmap_kwargs.get("width", 0)
        tick_params["rotation"] = heatmap_kwargs.get("rotation", 45)
        tick_params["labelsize"] = heatmap_kwargs.get("labelsize", "small")

        # ticks
        locater_x = plt.IndexLocator(1, 0.5)
        locater_y = plt.NullLocator()

        # cbar
        cbar_kwargs = cbar_kwargs or {}
        heatmap_kwargs["cbar_ax"] = axes["c"]
        heatmap_kwargs["cbar_kws"] = cbar_kwargs
        cbar_tick_params = tick_params.copy()
        cbar_title_kwargs = {
            "label": "Z-Score",
            "labelsize": tick_params["labelsize"],
        }

        heatmap(
            df_binned_piv=df_props_probe,
            ax=axes["h1"],
            locater_x=locater_x,
            locater_y=locater_y,
            formater_x=None,
            heatmap_kwargs=heatmap_kwargs,
            tick_params=tick_params,
            cbar_tick_params=cbar_tick_params,
            cbar_title_kwargs=cbar_title_kwargs,
        )
        heatmap(
            df_binned_piv=df_props_single_unit,
            ax=axes["h2"],
            locater_x=locater_x,
            formater_x=None,
            locater_y=locater_y,
            heatmap_kwargs=heatmap_kwargs,
            tick_params=tick_params,
            cbar_tick_params=cbar_tick_params,
            cbar_title_kwargs=cbar_title_kwargs,
        )
        axes["h1"].set_title("Silicon\nProbe")
        axes["h2"].set_title("Glass\nPipette")

        if textwrap_width is not None:
            for ax in (axes["h1"], axes["h2"]):
                wrap_all_text(ax, width=textwrap_width)
                ax.tick_params(**tick_params)

        return axes

    def plot_bars(
        self,
        df_props: pd.DataFrame,
        electrode_types_col: str = "Electrode Type",
        metrics_col: str = "Metric",
        value_col: str = "Value",
        neuron_type_col: str = "Neuron Type",
        fig: Optional[mpl.figure.Figure] = None,
        grid_spec_kwargs: Optional[Dict[str, Any]] = None,
        barplot_kwargs: Optional[Dict[str, Any]] = None,
        despine: bool = True,
        tick_params: Optional[Dict[str, Any]] = None,
        textwrap_width: Optional[int] = None,
    ):
        electrode_types = (
            df_props[electrode_types_col].value_counts().index.values.tolist()
        )
        metrics = df_props[metrics_col].value_counts().index.values.tolist()
        neuron_types = df_props[neuron_type_col].value_counts().index.values.tolist()
        # make axes grid
        fig = fig or plt.figure()
        grid_spec_kwargs = grid_spec_kwargs or {}
        grid_spec_kwargs["top"] = grid_spec_kwargs.get("top", 0.8)
        grid_spec_kwargs["bottom"] = grid_spec_kwargs.get("bottom", 0.1)
        grid_spec_kwargs["right"] = grid_spec_kwargs.get("right", 0.8)
        grid_spec_kwargs["left"] = grid_spec_kwargs.get("left", 0.25)
        grid_spec_kwargs["wspace"] = grid_spec_kwargs.get("wspace", 0.2)
        grid_spec_kwargs["wspace"] = grid_spec_kwargs.get("hspace", 0.4)

        axes = fig.subplots(
            nrows=len(metrics),
            ncols=len(electrode_types),
            sharey="row",
            sharex=True,
            gridspec_kw=grid_spec_kwargs,
        )

        # plot defaults
        barplot_kwargs = barplot_kwargs or {}
        barplot_kwargs["color"] = barplot_kwargs.get("color", "grey")
        barplot_kwargs["edgecolor"] = barplot_kwargs.get("edgecolor", "black")
        barplot_kwargs["linewidth"] = barplot_kwargs.get("linewidth", 0.8)
        barplot_kwargs["errwidth"] = barplot_kwargs.get("errwidth", 1.2)
        barplot_kwargs["capsize"] = barplot_kwargs.get("capsize", 0.15)
        barplot_kwargs["errcolor"] = barplot_kwargs.get("errcolor", "black")
        barplot_kwargs["alpha"] = barplot_kwargs.get("alpha", 1)
        barplot_kwargs["order"] = barplot_kwargs.get("order", neuron_types)

        # tick defaults
        tick_params = tick_params or {}

        for col_idx, electrode_type in enumerate(electrode_types):
            for row_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                df_ = df_props.loc[
                    lambda x: (x[electrode_types_col] == electrode_type)
                    & (x[metrics_col] == metric)
                ]
                sns.barplot(
                    data=df_,
                    x=neuron_type_col,
                    y=value_col,
                    ax=ax,
                    **barplot_kwargs,
                )
                if col_idx == 0:
                    ax.set_ylabel(metric)
                else:
                    ax.set_ylabel("")
                if row_idx == 0:
                    ax.set_title(electrode_type)
                else:
                    ax.set_title("")
                ax.set_xlabel("")
                if despine:
                    sns.despine(ax=ax)

                if textwrap_width is not None:
                    wrap_ylabel(ax, width=textwrap_width)

                ax.tick_params(**tick_params)
        fig.align_ylabels()
        return axes

    def plot_pct(
        self,
        df_props: pd.DataFrame,
        fig: Optional[mpl.figure.Figure] = None,
        grid_spec_kwargs: Optional[Dict[str, Any]] = None,
        neuron_type_col: str = "Neuron Type",
        electrode_types_col: str = "Electrode Type",
        ylabel: str = "% Total",
        palette: Optional[List[str]] = None,
        barplot_kwargs: Optional[Dict[str, Any]] = None,
        legend_fontsize: Optional[str] = None,
        despine: bool = True,
    ):
        palette = palette or PAL_GREY_BLACK
        electrode_types = (
            df_props[electrode_types_col].value_counts().index.values.tolist()
        )
        neuron_types = df_props[neuron_type_col].value_counts().index.values.tolist()
        fig = fig or plt.figure()
        grid_spec_kwargs = grid_spec_kwargs or {}
        grid_spec_kwargs["top"] = grid_spec_kwargs.get("top", 0.6)
        grid_spec_kwargs["bottom"] = grid_spec_kwargs.get("bottom", 0.3)
        grid_spec_kwargs["right"] = grid_spec_kwargs.get("right", 0.9)
        grid_spec_kwargs["left"] = grid_spec_kwargs.get("left", 0.1)
        ax = fig.subplots(gridspec_kw=grid_spec_kwargs)

        # plot defaults
        barplot_kwargs = barplot_kwargs or {}
        barplot_kwargs["edgecolor"] = barplot_kwargs.get("edgecolor", "black")
        barplot_kwargs["linewidth"] = barplot_kwargs.get("linewidth", 0.8)
        barplot_kwargs["alpha"] = barplot_kwargs.get("errwidth", 1)
        barplot_kwargs["order"] = barplot_kwargs.get("order", neuron_types)
        barplot_kwargs["hue_order"] = barplot_kwargs.get("hue_order", electrode_types)
        barplot_kwargs["palette"] = barplot_kwargs.get("palette", palette)

        ax = (
            df_props[neuron_type_col]
            .groupby(df_props[electrode_types_col])
            .value_counts(normalize=True)
            .rename(ylabel)
            .multiply(100)
            .reset_index()
            .pipe(
                (sns.barplot, "data"),
                x=neuron_type_col,
                y=ylabel,
                hue="Electrode Type",
                ax=ax,
                **barplot_kwargs,
            )
        )
        if legend_fontsize:
            plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)
        ax.set_xlabel("")
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=1,
            title=None,
            frameon=False,
        )
        if despine:
            sns.despine(ax=ax)
        return ax
