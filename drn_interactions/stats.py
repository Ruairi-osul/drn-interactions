from spiketimes.statistics import inter_spike_intervals, cv2
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
