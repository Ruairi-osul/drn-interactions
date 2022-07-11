from drn_interactions.transforms.nbox_transforms import align_to_data_by
from binit.bin import which_bin
import numpy as np
import warnings


class ShockUtils:
    """A container for a set of methods useful when working with foot shock data
    """

    def align_spikes(self, df_spikes, df_events, session=None):
        if session is not None:
            df_spikes = df_spikes.query("session_name == @session")
            df_events = df_events.query("session_name == @session")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = align_to_data_by(
                df_spikes,
                df_events,
                time_before_event=0.5,
                time_after_event=1.5,
                df_data_group_col="session_name",
                df_events_group_colname="session_name",
                df_events_timestamp_col="event_s",
                df_data_time_col="spiketimes",
            )
        return df

    def aligned_binned_from_spikes(
        self, df_spikes, df_events, session=None, bin_width=0.02
    ):
        df = self.align_spikes(df_spikes, df_events, session)
        bins = np.arange(-0.5, 1.5, bin_width)
        df["bin"] = np.round(which_bin(df["aligned"].values, bins), 2)
        df = (
            df.groupby(["neuron_id", "event", "bin"])
            .apply(len)
            .to_frame("counts")
            .reset_index()
        )
        return (
            df.pivot(index=["event", "bin"], columns="neuron_id", values="counts")
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
