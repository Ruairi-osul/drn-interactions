from drn_interactions.io import load_neurons_derived
from drn_interactions.config import Config
from .spikes import SpikesHandler
import numpy as np


def load_drug_data(block="chal", t_start=-600, t_stop=1200, bin_width=1):
    neurons = load_neurons_derived().query("session_name in @Config.drug_sessions")
    clusters = neurons[["neuron_id", "drug", "wf_3", "session_name"]]
    sessions = neurons["session_name"].unique().tolist()
    spikes = SpikesHandler(
        block=block,
        bin_width=bin_width,
        session_names=sessions,
        t_start=t_start,
        t_stop=t_stop,
    ).binned
    spikes = spikes.merge(clusters[["neuron_id", "drug"]])
    spikes["block"] = np.where(spikes["bin"] < 0, "pre", "post")
    return spikes, clusters
