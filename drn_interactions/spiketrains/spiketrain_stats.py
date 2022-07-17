from spiketimes.statistics import (
    inter_spike_intervals,
    cv2,
    mean_firing_rate,
    mean_firing_rate_ifr,
)
from spiketimes.binning import binned_spiketrain
import numpy as np
import pandas as pd


def cv_isi_burst(spiketrain, thresh=0.05) -> float:
    isi = inter_spike_intervals(spiketrain)
    isi = isi[isi > thresh]
    return cv2(isi)


def cv_isi(spiketrain, thresh=0.05):
    isi = inter_spike_intervals(spiketrain)
    return cv2(isi)


def num_bursts(spiketrain, thresh=0.05):
    isi = inter_spike_intervals(spiketrain)
    burst_isis = isi[(isi < thresh)]
    return len(burst_isis)


def is_burster(spiketrain, fraction_bursts=0.1, burst_thresh=0.05):
    frac = frac_bursts(spiketrain, burst_thresh=burst_thresh)
    return frac >= fraction_bursts


def frac_bursts(spiketrain, burst_thresh=0.05):
    n_tot = len(spiketrain)
    n_busts = num_bursts(spiketrain, thresh=burst_thresh)
    frac = n_busts / n_tot
    return frac


def median_burst_interval(spiketrain, thresh=0.01):
    isi = inter_spike_intervals(spiketrain)
    burst_isis = isi[(isi < thresh)]
    return np.median(burst_isis)


def max_firing_rate(spiketrain, bin_size=1, t_start=None, t_stop=None):
    _, counts = binned_spiketrain(
        spiketrain, fs=1 / bin_size, t_start=t_start, t_stop=t_stop
    )
    return np.max(counts)


def cv_firing_rate(spiketrain, bin_size=1, t_start=None, t_stop=None):
    _, counts = binned_spiketrain(
        spiketrain, fs=1 / bin_size, t_start=t_start, t_stop=t_stop
    )
    counts = counts + 0.0001  # to avoid divide by zero
    return cv2(counts)


class SpikeTrainStats:
    def __init__(self, thresh_burst=0.05, fraction_bursts=0.1):
        self.thresh_burst = thresh_burst
        self.fraction_bursts = fraction_bursts

    def __call__(self, spiketrain):
        out = {}
        out["cv_isi"] = cv_isi(spiketrain)
        out["cv_isi_burst"] = cv_isi_burst(spiketrain, thresh=self.thresh_burst)
        out["frac_bursts"] = frac_bursts(spiketrain, burst_thresh=self.thresh_burst)
        out["median_burst_interval"] = median_burst_interval(
            spiketrain, thresh=self.thresh_burst
        )
        out["is_burst"] = is_burster(
            spiketrain, burst_thresh=self.thresh_burst, fraction_bursts=self.fraction_bursts
        )
        out["mean_firing_rate"] = mean_firing_rate(spiketrain)
        return pd.Series(out)


class SpikeTrainDescriptor:
    def __init__(
        self, neuron_col: str = "neuron_id", spiketimes_col: str = "spiketimes"
    ):
        self.neuron_col = neuron_col
        self.spiketimes_col = spiketimes_col

    def cv2_isi_burst(self, df_spikes, burst_thresh=0.05):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            cv_isi_burst, thresh=burst_thresh
        )
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "cv2_isi_burst"}, inplace=True)
        return df_res

    def cv2_isi(self, df_spikes):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(cv_isi,)
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "cv2_isi"}, inplace=True)
        return df_res

    def is_buster(self, df_spikes, fraction_bursts=0.1, burst_thresh=0.05):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            is_burster, fraction_bursts=fraction_bursts, burst_thresh=burst_thresh
        )
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "is_buster"}, inplace=True)
        return df_res

    def median_burst_interval(self, df_spikes, thresh=0.01):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            median_burst_interval, thresh=thresh
        )
        df_res = df_res.reset_index()
        df_res.rename(
            columns={self.spiketimes_col: "median_burst_interval"}, inplace=True
        )
        return df_res

    def frac_bursts(self, df_spikes, burst_thresh=0.05):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            frac_bursts, burst_thresh=burst_thresh
        )
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "fraction_bursts"}, inplace=True)
        return df_res

    def mean_firing_rate(self, df_spikes, t_start=None, t_stop=None):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            mean_firing_rate, t_start=t_start, t_stop=t_stop
        )
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "mean_firing_rate"}, inplace=True)
        return df_res

    def mean_firing_rate_ifr(
        self, df_spikes, bin_size=1, exclude_below=None, t_start=None, t_stop=None
    ):
        fs = 1 / bin_size
        df_res = df_spikes.groupby("neuron_id")[self.spiketimes_col].apply(
            mean_firing_rate_ifr,
            fs=fs,
            exclude_below=exclude_below,
            t_start=t_start,
            t_stop=t_stop,
        )
        df_res = df_res.reset_index()
        df_res.rename(
            columns={self.spiketimes_col: "mean_firing_rate_ifr"}, inplace=True
        )

        return df_res

    def max_firing_rate(self, df_spikes, bin_size=1, t_start=None, t_stop=None):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            max_firing_rate, bin_size=bin_size, t_start=t_start, t_stop=t_stop
        )
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "max_firing_rate"}, inplace=True)
        return df_res

    def cv_firing_rate(self, df_spikes, bin_size=1, t_start=None, t_stop=None):
        df_res = df_spikes.groupby(self.neuron_col)[self.spiketimes_col].apply(
            cv_firing_rate, bin_size=bin_size, t_start=t_start, t_stop=t_stop
        )
        df_res = df_res.reset_index()
        df_res.rename(columns={self.spiketimes_col: "cv_firing_rate"}, inplace=True)
        return df_res

    def describe(
        self,
        df_spikes,
        t_start=None,
        t_stop=None,
        burst_thresh=0.05,
        fraction_bursts=0.1,
        mfr_bin_size=60,
        mfr_exclude_below=0.5,
        max_fr_binsize=60,
        cv_fr_binsize=60,
    ):

        df_res = self.cv2_isi(df_spikes)
        df_res = df_res.merge(
            self.is_buster(
                df_spikes, fraction_bursts=fraction_bursts, burst_thresh=burst_thresh
            ),
            on="neuron_id",
        )
        df_res = df_res.merge(
            self.cv2_isi_burst(df_spikes, burst_thresh=burst_thresh), on="neuron_id"
        )
        df_res = df_res.merge(
            self.median_burst_interval(df_spikes, thresh=burst_thresh), on="neuron_id"
        )
        df_res = df_res.merge(
            self.mean_firing_rate(df_spikes, t_start=t_start, t_stop=t_stop),
            on="neuron_id",
        )
        df_res = df_res.merge(
            self.mean_firing_rate_ifr(
                df_spikes,
                bin_size=mfr_bin_size,
                exclude_below=mfr_exclude_below,
                t_start=t_start,
                t_stop=t_stop,
            ),
            on="neuron_id",
        )
        df_res = df_res.merge(
            self.frac_bursts(df_spikes, burst_thresh=burst_thresh), on="neuron_id"
        )
        df_res = df_res.merge(
            self.max_firing_rate(
                df_spikes, bin_size=max_fr_binsize, t_start=t_start, t_stop=t_stop
            ),
            on="neuron_id",
        )
        df_res = df_res.merge(
            self.cv_firing_rate(
                df_spikes, bin_size=cv_fr_binsize, t_start=t_start, t_stop=t_stop
            ),
            on="neuron_id",
        )
        return df_res
