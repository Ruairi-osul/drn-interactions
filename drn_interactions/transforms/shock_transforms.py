from drn_interactions.transforms.nbox_transforms import align_to_data_by
from binit.bin import which_bin
import numpy as np
import warnings


class ShockUtils:
    """A container for a set of methods useful when working with foot shock data"""

    def __init__(self, session_col="session_name", event_time_col="event_s"):
        self.session_col = session_col
        self.event_time_col = event_time_col
        self.time_before_event = 0.5
        self.time_after_event = 1.5
        self.neuron_col = "neuron_id"

    def align_spikes(
        self,
        df_spikes,
        df_events,
        sessions=None,
        spikes_col="spiketimes",
    ):
        if sessions is not None:
            df_spikes = df_spikes.query(f"{self.session_name} in @sessions")
            df_events = df_events.query(f"{self.session_name} in @sessions")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = align_to_data_by(
                df_spikes,
                df_events,
                time_before_event=self.time_before_event,
                time_after_event=self.time_after_event,
                df_data_group_col=self.session_col,
                df_events_group_colname=self.session_col,
                df_events_timestamp_col=self.event_time_col,
                df_data_time_col=spikes_col,
            )
        return df

    def aligned_binned_from_spikes(
        self,
        df_spikes,
        df_events,
        sessions=None,
        bin_width=0.02,
    ):
        df = self.align_spikes(df_spikes, df_events, sessions)
        bins = np.arange(-1 * self.time_before_event, self.time_after_event, bin_width)
        df["bin"] = np.round(which_bin(df["aligned"].values, bins), 2)
        df = (
            df.groupby([self.neuron_col, "event", "bin"])
            .apply(len)
            .to_frame("counts")
            .reset_index()
        )
        return (
            df.pivot(index=["event", "bin"], columns=self.neuron_col, values="counts")
            .fillna(0)
            .reset_index()
        )

    def population_from_aligned_binned(self, df_aligned_binned):
        return (
            df_aligned_binned.melt(id_vars=["event", "bin"], var_name="neuron_id")
            .groupby(["event", "bin"], as_index=False)["value"]
            .mean()
        )

    def average_trace_from_aligned_binned(self, df_aligned_binned):
        return df_aligned_binned.drop("event", axis=1).groupby(["bin"]).mean()

    def average_population_from_population(self, df_population):
        return df_population.groupby("bin", as_index=False)["value"].mean()
