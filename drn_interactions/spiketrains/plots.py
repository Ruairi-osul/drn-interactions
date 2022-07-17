from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from drn_interactions.plots import heatmap
from sklearn.preprocessing import robust_scale
from drn_interactions.config import Config
from drn_interactions.io import load_waveforms
from drn_interactions.spiketrains.waveforms import WaveformPreprocessor


class NeuronTypesFigureLoader:
    def __init__(
        self,
        probe_neuron_types_path: Optional[Path] = None,
        probe_waveform_props_path: Optional[Path] = None,
        probe_spiketrain_props_path: Optional[Path] = None,
        single_unit_props_path: Optional[Path] = None,
        single_unit_neuron_types_path: Optional[Path] = None,
    ):
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

    def __call__(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        probe_neuron_types = pd.read_csv(self.probe_neuron_types_path)
        probe_waveforms = load_waveforms()
        probe_waveform_props = pd.read_csv(self.probe_waveform_props_path)
        probe_spiketrain_props = pd.read_csv(self.probe_spiketrain_props_path)
        single_unit_props = pd.read_csv(self.single_unit_props_path)
        single_unit_neuron_types = pd.read_csv(self.single_unit_neuron_types_path)
        return (
            probe_neuron_types,
            probe_waveforms,
            probe_waveform_props,
            probe_spiketrain_props,
            single_unit_props,
            single_unit_neuron_types,
        )


class NeuronTypesFigurePreprocessor:
    def __init__(
        self,
        electrode_type_order: Sequence[str] = ("Silicon Probe", "Glass Pipette"),
        neuron_type_order: Sequence[str] = ("SR", "SIR", "FF"),
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
        self.meta_cols = [c for c in new_col_names if c not in metrics]
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

    def preprocess_waveforms(self, probe_waveforms: pd.DataFrame) -> pd.DataFrame:
        return self.waveform_preprocessor(probe_waveforms)

    def __call__(
        self,
        probe_neuron_types: pd.DataFrame,
        probe_waveforms: pd.DataFrame,
        probe_waveform_props: pd.DataFrame,
        probe_spiketrain_props: pd.DataFrame,
        single_unit_props: pd.DataFrame,
        single_unit_neuron_types: pd.DataFrame,
    ):
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
        df_waveforms = self.preprocess_waveforms(probe_waveforms)
        return df, df_waveforms


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
    def __init__(
        self,
        figsize: Tuple[float, float] = (7, 7),
        constrained_layout: bool = True,
        top_bottom_hspace: float = 0.3,
        top_bottom_height_ratios: Tuple[float, float] = (0.5, 0.5),
        raster_waveforms_width_ratios: Tuple[float, float] = (0.9, 0.3),
        raster_waveforms_wspace: float = 0.01,
        bottom_figs_width_ratios: Tuple[float, float, float] = (0.5, 0.4, 0.2),
        bottom_figs_wspace: float = 0.2,
        raster_left: float = 0.05,
        raster_right: float = 1,
        raster_bottom: float = 0.1,
        raster_top: float = 0.9,
        raster_hspace: float = 0.1,
        pct_top: float = 0.6,
        pct_bottom: float = 0.3,
        pct_left: float = 0.3,
        pct_right: float = 0.8,
        bar_wspave: float = 0.4,
        bar_hspace: float = 0.4,
        bar_top: float = 0.9,
        bar_bottom: float = 0.1,
        bar_right: float = 0.8,
    ):
        self.figsize = figsize
        self.constrained_layout = constrained_layout
        self.top_bottom_hspace = top_bottom_hspace
        self.top_bottom_height_ratios = top_bottom_height_ratios
        self.raster_waveforms_width_ratios = raster_waveforms_width_ratios
        self.raster_waveforms_wspace = raster_waveforms_wspace
        self.raster_left = raster_left
        self.raster_right = raster_right
        self.raster_bottom = raster_bottom
        self.raster_top = raster_top
        self.raster_hspace = raster_hspace
        self.pct_top = pct_top
        self.pct_bottom = pct_bottom
        self.pct_left = pct_left
        self.pct_right = pct_right
        self.bar_wspave = bar_wspave
        self.bar_hspace = bar_hspace
        self.bar_top = bar_top
        self.bar_bottom = bar_bottom
        self.bar_right = bar_right
        self.bottom_figs_width_ratios = bottom_figs_width_ratios
        self.bottom_figs_wspace = bottom_figs_wspace

    def __call__(self):
        out = {}
        fig = plt.figure(
            figsize=self.figsize, constrained_layout=self.constrained_layout
        )
        f_top, f_bottom = fig.subfigures(
            nrows=2,
            height_ratios=self.top_bottom_height_ratios,
            hspace=self.top_bottom_hspace,
        )
        f_raster, f_waveforms = f_top.subfigures(
            ncols=2,
            nrows=1,
            width_ratios=self.raster_waveforms_width_ratios,
            wspace=self.raster_waveforms_wspace,
        )
        f_heat, f_bar, f_prop = f_bottom.subfigures(
            ncols=3,
            width_ratios=self.bottom_figs_width_ratios,
            wspace=self.bottom_figs_wspace,
        )
