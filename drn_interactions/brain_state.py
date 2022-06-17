from binit.bin import which_bin
import numpy as np


def get_state_piv(
    spikes, eeg, state_col="state", index_name="bin", eeg_time_col="timepoint_s"
):
    spikes = spikes.copy()
    return (
        spikes.reset_index()
        .assign(
            eeg_time=lambda x: which_bin(
                x[index_name].values,
                eeg[eeg_time_col].values,
                time_before=0,
                time_after=2,
            )
        )
        .merge(eeg, left_on="eeg_time", right_on=eeg_time_col)
        .set_index(index_name)[list(spikes.columns) + [state_col]]
    )


def get_state_long(spikes, eeg, index_name="bin", eeg_time_col="timepoint_s"):
    return (
        spikes.reset_index()
        .copy()
        .assign(
            eeg_bin=lambda x: which_bin(
                x[index_name].values,
                eeg[eeg_time_col].values,
                time_before=0,
                time_after=2,
            )
        )
        .merge(eeg, left_on="eeg_bin", right_on="timepoint_s")
        .drop("timepoint_s", axis=1)
    )


def most_common_state_in_segment(spikes, eeg, segment_col="segment", state_col="state"):
    df = get_state_long(spikes, eeg, index_name="spiketimes")
    return (
        df.groupby(segment_col)[state_col]
        .apply(lambda x: (x == "sw").mean())
        .to_frame("prop_sw")
        .reset_index()
        .assign(state=lambda x: np.where(x.prop_sw >= 0.5, "sw", "act"))
        .merge(df.drop("state", axis=1))
    )
