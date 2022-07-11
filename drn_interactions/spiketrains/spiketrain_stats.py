from spiketimes.statistics import inter_spike_intervals, cv2, mean_firing_rate
import numpy as np


def cv_isi_burst(spiketrain, thresh=0.05) -> float:
    isi = inter_spike_intervals(spiketrain)
    isi = isi[isi > thresh]
    return cv2(isi)


def cv_isi(spiketrain, thresh=0.05):
    isi = inter_spike_intervals(spiketrain)
    isi = isi[isi > thresh]
    return cv2(isi)


def num_bursts(spiketrain, thresh=0.05):
    isi = inter_spike_intervals(spiketrain)
    burst_isis = isi[(isi < thresh)]
    return len(burst_isis)


def is_burster(spiketrain, frac_bursts=0.1, burst_thresh=0.05):
    n_tot = len(spiketrain)
    n_busts = num_bursts(spiketrain, thresh=burst_thresh)
    frac = n_busts / n_tot
    return frac >= frac_bursts


def median_burst_interval(spiketrain, thresh=0.01):
    isi = inter_spike_intervals(spiketrain)
    burst_isis = isi[(isi < thresh)]
    return np.median(burst_isis)


class SpikeTrainStats:
    def __init__(self, burst_thresh=0.05, fraction_bursts=0.1):
        self.burst_thresh = burst_thresh
        self.fraction_bursts = fraction_bursts

    def cv2_isi(self, df_spikes, burst_thresh=0.05):
        df_res = df_spikes.groupby("neuron_id")["spiketimes"].apply(
            cv_isi_burst, thresh=burst_thresh
        )
        df_res = df_res.reset_index()
        return df_res

    def is_buster(self, df_spikes, frac_bursts=0.1, burst_thresh=0.05):
        df_res = df_spikes.groupby("neuron_id")["spiketimes"].apply(
            is_burster, frac_bursts=frac_bursts, burst_thresh=burst_thresh
        )
        df_res = df_res.reset_index()
        return df_res

    def median_burst_interval(self, df_spikes, thresh=0.01):
        df_res = df_spikes.groupby("neuron_id")["spiketimes"].apply(
            median_burst_interval, thresh=thresh
        )
        df_res = df_res.reset_index()
        return df_res

    def mean_firing_rate(
        self,
        df_spikes,
    ):
        df_res = df_spikes.groupby("neuron_id")["spiketimes"].apply(mean_firing_rate)
        df_res = df_res.reset_index()
        return df_res
