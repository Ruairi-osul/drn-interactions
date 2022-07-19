from .spikes import SpikesHandler
from .brain_state import RawEEGHandler, StateHandler, align_to_state
from drn_interactions.io import load_neurons
from neurobox.long_transforms import get_closest_event
import warnings
import pandas as pd
from typing import Optional


def align_bins_to_states_long(
    spikes_handler: SpikesHandler,
    states_handler: StateHandler,
    neuron_types: Optional[pd.DataFrame],
) -> pd.DataFrame:
    spikes = spikes_handler.binned.copy()
    meta = load_neurons()[["neuron_id", "session_name", "group_name"]]
    spikes = meta.merge(spikes, left_on="neuron_id", right_on="neuron_id")
    states = states_handler.states_df.copy()
    time_after = states[states_handler.time_column].diff().values[0]

    df_aligned = align_to_state(
        df_data=spikes,
        df_state=states,
        time_after=time_after,
        df_state_session_col=states_handler.session_column,
        df_state_time_col=states_handler.time_column,
        df_data_time_col="bin",
        df_data_session_col="session_name",
    )
    if neuron_types is not None:
        df_aligned = df_aligned.merge(
            neuron_types,
        )
    return df_aligned


def align_spikes_to_states_long(
    spikes_handler: SpikesHandler,
    states_handler: StateHandler,
    neuron_types: Optional[pd.DataFrame],
) -> pd.DataFrame:
    spikes = spikes_handler.spikes.copy()
    meta = load_neurons()[["neuron_id", "session_name", "group_name"]]
    spikes = meta.merge(spikes, left_on="neuron_id", right_on="neuron_id")
    states = states_handler.states_df.copy()
    time_after = states[states_handler.time_column].diff().values[0]

    df_aligned = align_to_state(
        df_data=spikes,
        df_state=states,
        time_after=time_after,
        df_state_session_col=states_handler.session_column,
        df_state_time_col=states_handler.time_column,
        df_data_time_col="spiketimes",
        df_data_session_col="session_name",
    )
    if neuron_types is not None:
        df_aligned = df_aligned.merge(
            neuron_types,
        )
    return df_aligned


def align_spikes_to_phase_long(
    spikes_handler: SpikesHandler,
    raw_eeg_handler: RawEEGHandler,
    states_handler: StateHandler,
    neuron_types: Optional[pd.DataFrame],
    t_before: float = 0,
    t_after: float = 0.1,
) -> pd.DataFrame:

    spikes_with_state = align_spikes_to_states_long(
        spikes_handler=spikes_handler,
        states_handler=states_handler,
        neuron_types=neuron_types,
    )
    eeg = raw_eeg_handler.raw_eeg_df

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df_aligned = get_closest_event(
            df_data=spikes_with_state,
            df_events=eeg,
            time_after=t_after,
            time_before=t_before,
            df_data_group_colname="session_name",
            df_events_group_colname=raw_eeg_handler.session_column,
            df_events_timestamp_col=raw_eeg_handler.time_column,
            df_data_time_col="spiketimes",
            returned_colname="raw_eeg_bin",
        )
    df_aligned = df_aligned.dropna()
    df_aligned = df_aligned.merge(
        eeg,
        left_on=["session_name", "raw_eeg_bin"],
        right_on=[raw_eeg_handler.session_column, raw_eeg_handler.time_column],
    )

    return df_aligned
