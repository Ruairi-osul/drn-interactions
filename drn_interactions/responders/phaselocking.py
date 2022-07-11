import pandas as pd
from drn_interactions.transforms import SpikesHandler
from drn_interactions.load import (
    load_derived_generic,
    load_eeg_raw,
    load_neurons_derived,
)
from typing import Tuple



def load_phaselock_data(
    block: str = "pre",
    state_quality_filter: bool = True,
    t_start: float = 0,
    t_stop: float = 1800,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_eeg = load_eeg_raw(block).rename(columns={"timepoint_s": "time"})
    eeg_states = load_derived_generic("eeg_states.csv").rename(
        columns={"cluster": "state"}
    )
    if state_quality_filter:
        eeg_states = eeg_states.query("quality == 'good'").copy()

    neurons = load_neurons_derived().merge(
        eeg_states[["session_name"]].drop_duplicates()
    )
    sessions = neurons[["session_name"]].drop_duplicates()

    df_eeg = df_eeg.merge(sessions).query("time > @t_start and time < @t_stop")

    eeg_states = eeg_states.merge(sessions).query(
        "timepoint_s > @t_start and timepoint_s < @t_stop"
    )
    sh = SpikesHandler(
        block="pre",
        bin_width=0.1,
        session_names=sessions["session_name"].unique().tolist(),
        t_start=t_start,
        t_stop=t_stop,
    )
    spikes = sh.spikes.copy()
    spikes = spikes.merge(neurons[["neuron_id", "session_name"]])
    spikes = spikes[["neuron_id", "session_name", "spiketimes"]]
    return neurons, spikes, df_eeg, eeg_states


