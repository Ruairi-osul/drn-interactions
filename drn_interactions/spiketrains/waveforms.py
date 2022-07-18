from scipy.ndimage import gaussian_filter1d, median_filter
from typing import Any, Callable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.base import TransformerMixin
import seaborn as sns
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
from sklearn.ensemble import IsolationForest
from typing import Tuple, Union
from scipy.signal import peak_widths, peak_prominences
from umap import UMAP


class WaveformPreprocessor:
    def __init__(
        self,
        gaussian_sigma: Optional[float] = None,
        medfilter_size: Optional[int] = None,
        scale_to_baseline: bool = True,
        baseline_start_idx: int = 0,
        baseline_end_idx: int = 60,
        minmax: bool = True,
    ):
        self.gaussian_sigma = gaussian_sigma
        self.medfilter_size = medfilter_size
        self.scale_to_baseline = scale_to_baseline
        self.baseline_start_idx = baseline_start_idx
        self.baseline_end_idx = baseline_end_idx
        self.minmax = minmax
        self.baseline_scaler_fac = lambda: StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    @staticmethod
    def pivot(
        df_waveforms: pd.DataFrame, neuron_col: str, time_col: str, value_col: str
    ) -> pd.DataFrame:
        return df_waveforms.pivot(index=neuron_col, columns=time_col, values=value_col)

    @staticmethod
    def scale_to_pre(
        vals: pd.Series,
        baseline_start_idx: int = 0,
        baseline_end_idx: int = 60,
        transformer_fac=Callable[[Any], TransformerMixin],
    ):
        vals = vals.values.reshape(-1, 1)
        transformer = transformer_fac()
        transformer.fit(vals[baseline_start_idx:baseline_end_idx])
        return transformer.transform(vals).flatten()

    def _med_filter(self, vals: np.ndarray) -> np.ndarray:
        return median_filter(vals, size=self.medfilter_size)

    def __call__(
        self,
        df_waveforms: pd.DataFrame,
        neuron_col: str = "neuron_id",
        time_col: str = "waveform_index",
        value_col: str = "waveform_value",
    ) -> pd.DataFrame:
        df_piv = self.pivot(df_waveforms, neuron_col, time_col, value_col)
        if self.medfilter_size is not None:
            df_piv = df_piv.transform(
                self._med_filter,
                axis=1,
            )

        if self.gaussian_sigma is not None:
            df_piv = df_piv.transform(
                gaussian_filter1d,
                sigma=self.gaussian_sigma,
                axis=1,
            )

        if self.scale_to_baseline:
            df_piv = df_piv.transform(
                self.scale_to_pre,
                baseline_start_idx=self.baseline_start_idx,
                baseline_end_idx=self.baseline_end_idx,
                transformer_fac=self.baseline_scaler_fac,
                axis=1,
            )
        if self.minmax:
            df_piv = df_piv.transform(minmax_scale, axis=1)
        return df_piv


class WaveformOutliers:
    def __init__(
        self,
        n_estimators=100,
        contamination="auto",
        max_features=10,
        max_samples="auto",
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.mod = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
        )

    def detect_outliers(self, df_waveforms_piv: pd.DataFrame) -> None:
        X = df_waveforms_piv.values
        self.preds = self.mod.fit_predict(X)

    def subset_df(
        self, df_waveforms_piv: pd.DataFrame, preds: np.ndarray
    ) -> pd.DataFrame:
        self.outliers = self.neurons[preds == -1]
        self.good_units = self.neurons[preds != -1]
        df_good = df_waveforms_piv.loc[self.good_units]
        df_outiers = df_waveforms_piv.loc[self.outliers]
        return df_good, df_outiers

    def __call__(self, df_waveforms_piv: pd.DataFrame) -> pd.DataFrame:
        self.neurons = df_waveforms_piv.index.values
        self.detect_outliers(df_waveforms_piv)
        df_good, df_outliers = self.subset_df(df_waveforms_piv, self.preds)
        return df_good, df_outliers


class WaveformError(Exception):
    pass


class WaveformPeaks:
    def __init__(
        self,
        tol_min_peak: float = 0.4,
        prominence_min: float = 0.15,
        prominence_basepre: float = 0.05,
        prominence_basepost: float = 0.05,
        base_rel_height: float = 0.8,
        width_rel_height: float = 0.5,
    ):
        self.tol_min_peak = tol_min_peak
        self.prominence_min = prominence_min
        self.prominence_basepre = prominence_basepre
        self.prominence_basepost = prominence_basepost
        self.base_rel_height = base_rel_height
        self.width_rel_height = width_rel_height

    def find_peaks(
        self,
        waveform: np.ndarray,
        tol_min_peak: float = 0.4,
        prominence_min: float = 0.15,
        prominence_basepre: float = 0.05,
        prominence_basepost: float = 0.05,
    ) -> Tuple[int, int, int]:
        try:
            peak_min = self._find_peak_min(
                waveform, tolerance=tol_min_peak, prominence=prominence_min
            )
        except WaveformError as e:
            return np.nan, np.nan, np.nan  # return find_peaks(waveform, tol=tol)
        peak_base_pre = self._find_peak_basepre(
            waveform, peak_min, prominence=prominence_basepre
        )
        peak_base_post = self._find_peak_basepost(
            waveform, peak_min, prominence=prominence_basepost
        )
        return peak_base_pre, peak_min, peak_base_post

    def peak_metrics(
        self,
        waveform: np.ndarray,
        peak_base_pre: int,
        peak_min: int,
        peak_base_post: int,
        base_rel_height: float = 0.8,
        width_rel_height: float = 0.5,
    ) -> pd.DataFrame:
        out = {}
        to_iter = zip(
            ("min", "basepre", "basepost"),
            (waveform * -1, waveform, waveform),
            (peak_min, peak_base_pre, peak_base_post),
        )
        for name, waveform, peak_idx in to_iter:
            has_peak = peak_idx is not np.nan

            peak_width = self._peak_width(
                waveform, peak_idx, rel_height=width_rel_height
            )
            peak_prom = self._peak_prom(waveform, peak_idx)

            peak_height = self._peak_height(waveform, peak_idx)
            base_l, base_r = self._peak_bases(
                waveform, peak_idx, rel_height=base_rel_height
            )
            out[name] = dict(
                has_peak=has_peak,
                peak_width=peak_width,
                peak_prom=peak_prom,
                peak_height=peak_height,
                base_l=base_l,
                base_r=base_r,
            )

        df_out = pd.DataFrame(out)
        df_out = df_out.reset_index().rename(columns={"index": "metric"})
        df_out = df_out.melt(id_vars=["metric"], var_name="peak")
        return df_out

    def summarize_peaks(self, df_metrics: pd.DataFrame) -> pd.Series:
        out = {}
        out["width_min"] = df_metrics.query("peak == 'min' and metric == 'peak_width'")[
            "value"
        ].item()
        out["width_basepost"] = df_metrics.query(
            "peak == 'basepost' and metric == 'peak_width'"
        )["value"].item()
        out["prom_min"] = df_metrics.query("peak == 'min' and metric == 'peak_prom'")[
            "value"
        ].item()
        out["prom_basepre"] = df_metrics.query(
            "peak == 'basepre' and metric == 'peak_prom'"
        )["value"].item()
        out["prom_basepost"] = df_metrics.query(
            "peak == 'basepost' and metric == 'peak_prom'"
        )["value"].item()

        out["has_basepre"] = df_metrics.query(
            "peak == 'basepre' and metric == 'has_peak'"
        )["value"].item()

        out["basepost_width"] = df_metrics.query(
            "peak == 'basepost' and metric == 'peak_width'"
        )["value"].item()

        out["min_init"] = df_metrics.query("peak == 'min' and metric == 'base_l'")[
            "value"
        ].item()
        out["basepre_init"] = df_metrics.query(
            "peak == 'basepre' and metric == 'base_l'"
        )["value"].item()

        out["min_term"] = df_metrics.query("peak == 'min' and metric == 'base_r'")[
            "value"
        ].item()
        out["basepost_term"] = df_metrics.query(
            "peak == 'basepost' and metric == 'base_r'"
        )["value"].item()

        out["width_all"] = out["basepost_term"] - out["basepre_init"]
        out["width_premin"] = out["min_term"] - out["basepre_init"]
        out["width_minpost"] = out["basepost_term"] - out["min_term"]

        out["init"] = (
            out["basepre_init"]
            if not np.isnan(out["basepre_init"])
            else out["min_init"]
        )
        out["term"] = (
            out["basepost_term"]
            if not np.isnan(out["basepost_term"])
            else out["min_term"]
        )
        out["width_overall"] = out["term"] - out["init"]

        return pd.Series(out)

    def __call__(self, waveform: np.ndarray, summarize: bool = True) -> pd.Series:
        peak_base_pre, peak_min, peak_base_post = self.find_peaks(
            waveform,
            tol_min_peak=self.tol_min_peak,
            prominence_min=self.prominence_min,
            prominence_basepre=self.prominence_basepre,
            prominence_basepost=self.prominence_basepost,
        )
        df_metrics = self.peak_metrics(
            waveform,
            peak_base_pre=peak_base_pre,
            peak_min=peak_min,
            peak_base_post=peak_base_post,
            base_rel_height=self.base_rel_height,
            width_rel_height=self.width_rel_height,
        )
        if summarize:
            df_metrics = self.summarize_peaks(df_metrics)
        return df_metrics

    def _find_peak_min(
        self, waveform: np.ndarray, tolerance: float = 0.3, prominence: float = 0.15
    ) -> int:
        peak_min_idx, _ = find_peaks(
            waveform * -1,
            prominence=prominence,
        )
        is_min = self._peak_is_min(waveform, peak_min_idx, tolerance=tolerance)
        min_peak = peak_min_idx[is_min]
        try:
            min_peak = min_peak.item()
        except ValueError as e:
            raise WaveformError("Multiple negative peaks found")
        return min_peak

    def _peak_is_min(self, waveform: np.ndarray, peak_idx: int, tolerance: float = 0.3):
        num_samples = len(waveform)
        center_offset = self._distance_from_center(waveform, peak_idx)
        return center_offset < (num_samples * tolerance)

    def _find_peak_basepre(
        self, waveform: np.ndarray, peak_min: int, prominence: float = 0.05
    ) -> int:
        peak_max_idx, _ = find_peaks(
            waveform,
            prominence=prominence,
        )
        try:
            pre_min_peaks = peak_max_idx[peak_max_idx < peak_min]
            last_peak_before_min = pre_min_peaks[-1]
        except IndexError:
            last_peak_before_min = np.nan
        return last_peak_before_min

    def _find_peak_basepost(
        self, waveform: np.ndarray, peak_min: int, prominence: float = 0.05
    ) -> int:
        peak_max_idx, _ = find_peaks(
            waveform,
            prominence=prominence,
        )
        try:
            pre_min_peaks = peak_max_idx[peak_max_idx > peak_min]
            first_peak_after_min = pre_min_peaks[0]
        except IndexError:
            first_peak_after_min = np.nan
        return first_peak_after_min

    @staticmethod
    def _distance_from_center(waveform: np.ndarray, peak_idx: int):
        num_samples = len(waveform)
        center_idx = int(num_samples / 2)
        return abs(peak_idx - center_idx)

    @staticmethod
    def _peak_width(waveform: np.ndarray, peak_idx: int, rel_height=0.5) -> int:
        if np.isnan(peak_idx):
            return np.nan
        widths, *_ = peak_widths(waveform, peaks=[peak_idx], rel_height=rel_height)
        return widths.item()

    @staticmethod
    def _peak_prom(waveform: np.ndarray, peak_idx: int) -> float:
        if np.isnan(peak_idx):
            return np.nan
        prom, *_ = peak_prominences(waveform, peaks=[peak_idx])
        return prom.item()

    @staticmethod
    def _peak_height(waveform: np.ndarray, peak_idx: int) -> float:
        if np.isnan(peak_idx):
            return np.nan
        _, heights, *_ = peak_widths(waveform, peaks=[peak_idx])
        return heights.item()

    @staticmethod
    def _peak_bases(
        waveform: np.ndarray, peak_idx: int, rel_height=0.8
    ) -> Tuple[float, float]:
        if np.isnan(peak_idx):
            return np.nan, np.nan
        _, *_, left, right = peak_widths(
            waveform, peaks=[peak_idx], rel_height=rel_height
        )
        return left.item(), right.item()


class WaveMap:
    def __init__(self, **umap_kwargs):
        self.umap_kwargs = umap_kwargs
        self.mod: UMAP = UMAP(**umap_kwargs)

    def __call__(self, df_waveforms: pd.DataFrame) -> pd.DataFrame:
        self.mod.fit(df_waveforms.values)
        df = pd.DataFrame(
            self.mod.embedding_,
            columns=[f"UMAP-{i}" for i in range(1, self.mod.n_components + 1)],
            index=df_waveforms.index,
        )
        return df
