import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean


# def peak_asymmetry_by_neuron(
#     df: pd.core.frame.DataFrame,
#     peak_names: list = None,
#     neuron_col: str = "neuron_id",
#     peak_value_col: str = "peak_idx",
#     peak_name_col: str = "peak_name",
# ):
#     """
#     Given a dataframe containing a peaks of a single average waveform,
#     caluclates its peak assymetry
#     Asymmertry:
#         (B - A) / (A + B)
#     params:
#         df: df of waveform peaks
#         peak_names: the names of the peaks contained in peak_name_col for
#                     use in the calculation
#         neuron_col: label of the column containing neuron ids
#         peak_name_col: label of the column containg peak names
#         peak_value_col: label of the column containing peak values
#     returns:
#         pd.DataFrame containg columns {neuron_id, peak_asymmetry}
#     """
#     return (
#         df.groupby(neuron_col)
#         .apply(
#             lambda x: calculate_peak_asymmetry(
#                 x,
#                 peak_names=peak_names,
#                 peak_value_col=peak_value_col,
#                 peak_name_col=peak_name_col,
#             )
#         )
#         .reset_index()
#         .rename(columns={0: "peak_asymmetry"})
#     )


def calculate_peak_asymmetry(
    df: pd.core.frame.DataFrame,
    peak_names: list = None,
    peak_value_col: str = "peak_idx",
    peak_name_col: str = "peak_name",
):
    """
    Given a dataframe containing a peaks of a single average waveform,
    caluclates its peak assymetry
    Asymmertry:
        (B - A) / (A + B)
    params:
        df: df of waveform peaks
        peak_names: the names of the peaks contained in peak_name_col for
                    use in the calculation
        peak_name_col: label of the column containg peak names
        peak_value_col: label of the column containing peak values
    returns:
        float corresponding to the asymmetry
    """
    if peak_names is None:
        peak_names = ["initiation", "ahp"]

    first_value = df[df[peak_name_col] == peak_names[0]][peak_value_col].values[0]
    second_value = df[df[peak_name_col] == peak_names[1]][peak_value_col].values[0]

    return (second_value - first_value) / (first_value + second_value)


# def waveform_width_by_neuron(
#     df: pd.core.frame.DataFrame,
#     peak_names: list = None,
#     neuron_col: str = "neuron_id",
#     peak_name_col: str = "peak_name",
#     peak_idx_col: str = "peak_idx",
# ):
#     """
#     Given a dataframe with one row per wavefrom peak, calculates waveform width
#     for each neuron.
#     params:
#         df: dataframe containing the data
#         peak_names: names of peaks to be used to calculate distance. Should be a
#                     list of strings
#         neuron_col: label of column identifying the neuron responsible for the
#                     waveform peak
#         peak_name_col: label of column identifying the peak name
#         peak_idx_col: label of column containing the index position
#                        of timing of the peak
#     returns:
#         the euclidean distance between the two peaks
#     """
#     return (
#         df.groupby(neuron_col)
#         .apply(
#             lambda x: calculate_waveform_width(
#                 x,
#                 peak_names=peak_names,
#                 peak_name_col=peak_name_col,
#                 peak_idx_col=peak_idx_col,
#             )
#         )
#         .reset_index()
#         .rename(columns={0: "waveform_width"})
#     )


def calculate_waveform_width(
    df: pd.core.frame.DataFrame,
    peak_names: list = None,
    peak_name_col: str = "peak_name",
    peak_idx_col: str = "peak_idx",
):
    """
    Given a dataframe with one row per wavefrom peak, calculates waveform width
    params:
        df: dataframe containing the data
        peak_names: names of peaks to be used to calculate distance. Should be a
                    list of strings
        peak_name_col: label of column identifying the peak name
        peak_idx_col: label of column containing the index position
                       of timing of the peak
    returns:
        the euclidean distance between the two peaks
    """
    if df[peak_idx_col].isna().any():
        raise ValueError("Cannot run on data with NaNs. Use df.dropna() first")
    if peak_names is None:
        peak_names = ["minimum", "ahp"]
    first_value = df[df[peak_name_col] == peak_names[0]][peak_idx_col].values
    second_value = df[df[peak_name_col] == peak_names[1]][peak_idx_col].values
    assert len(first_value) == 1
    assert len(second_value) == 1
    return euclidean(first_value, second_value)


# def waveform_peaks_by_neuron(
#     df: pd.core.frame.DataFrame,
#     neuron_col: str = "neuron_id",
#     index_col: str = "waveform_index",
#     value_col: str = "waveform_value",
#     range_around: int = 70,
#     sigma: float = 1,
#     diff_threshold: float = 25,
# ):
#     """
#     Given a dataframe containing timepoints of average waveforms from multiple neurons,
#     calculates the spike initation, minimum and after spike hyper polerisation
#     for each waveform.
#     params:
#         neuron_col: column label for column containing neuron ids
#         value_col: column label for column containing waveform values
#         index_col: column label for column containing waveform index
#         diff_col: column label for column containing the differentiated values (col.diff())
#         range_aroud: hyperparameter controling maximum range around peak to search for
#                      initialisation and ahp peaks
#         sigma: hyper parameter controling gaussian smoothing during peak finding
#         diff_threshold: hyperparameter controling minimum slope required for the minimum peak.
#                         inscrease to require larger slope.

#     returns:
#         - df containing columns: {neuron_id, peak_name, peak_idx, peak_value}
#           peak_idx refers to the index of the peak on the average waveform.
#     """
#     return (
#         df.groupby(neuron_col)
#         .apply(
#             _get_waveform_peaks_by_neuron,
#             index_col=index_col,
#             value_col=value_col,
#             range_around=range_around,
#             sigma=sigma,
#             diff_threshold=diff_threshold,
#         )
#         .reset_index()
#         .drop("level_1", axis=1)
#     )


# def _get_waveform_peaks_by_neuron(
#     df: pd.core.frame.DataFrame,
#     index_col: str = "waveform_index",
#     value_col: str = "waveform_value",
#     range_around: int = 70,
#     sigma: float = 1,
#     diff_threshold: float = 25,
# ):
#     """
#     Given a group by object derived from a dataframe containing timepoints of
#     average waveforms, calculates the spike initation, minimum and
#     after spike hyper polerisation for each waveform
#     params
#         value_col: column label for column containing waveform values
#         index_col: column label for column containing waveform index
#         diff_col: column label for column containing the differentiated values (col.diff())
#         range_aroud: hyperparameter controling maximum range around peak to search for
#                      initialisation and ahp peaks
#         sigma: hyper parameter controling gaussian smoothing during peak finding
#         diff_threshold: hyperparameter controling minimum slope required for the minimum peak.
#                         inscrease to require larger slope.
#     """
#     return (
#         df.sort_values(by=["waveform_index"])
#         .pipe(lambda x: x.assign(filt=gaussian_filter1d(x[value_col], sigma=sigma)))
#         .pipe(lambda x: x.assign(diff_=x["filt"].diff()))
#         .pipe(
#             lambda x: _find_waveform_peaks(
#                 x,
#                 value_col="filt",
#                 index_col=index_col,
#                 range_around=range_around,
#                 diff_col="diff_",
#                 diff_threshold=diff_threshold,
#             )
#         )
#     )


def find_waveform_peaks(
    df: pd.core.frame.DataFrame,
    value_col: str = "filt",
    index_col: str = "waveform_index",
    diff_col: str = "diff_",
    range_around: int = 70,
    diff_threshold: float = 25,
):
    """
    Given a dataframe containing timepoints of one average waveform,
    calculates the spike initation, minimum and after spike hyper polerisation
    params
        value_col: column label for column containing waveform values
        index_col: column label for column containing waveform index
        diff_col: column label for column containing the differentiated values (col.diff())
        range_aroud: hyperparameter controling maximum range around peak to search for
                     initialisation and ahp peaks
        diff_threshold: hyperparameter controling minimum slope required for the minimum peak.
                        inscrease to require larger slope.
    """

    min_val = df[value_col].min()
    min_idx = df[df[value_col] == min_val][index_col].values[0]

    try:
        # TODO implement warning
        before_val = (
            df[
                (df[index_col] > (min_idx - range_around))
                & (df[index_col] < min_idx)
                & (
                    np.absolute(df[diff_col])
                    > (diff_threshold * np.absolute(np.nanmedian(df[diff_col])))
                )
            ]
        )[value_col].max()

        before_idx = df[df[value_col] == before_val][index_col].values[0]

    except IndexError:
        return pd.DataFrame(
            {
                "peak_name": ["initiation", "minimum", "ahp"],
                "peak_idx": [np.nan, np.nan, np.nan],
                "peak_value": [np.nan, np.nan, np.nan],
            }
        )

    after_val = df[
        (df[index_col] < (min_idx + range_around)) & (df[index_col] > (min_idx))
    ][value_col].max()

    try:
        after_idx = df[df[value_col] == after_val][index_col].values[0]
    except IndexError:
        return pd.DataFrame(
            {
                "peak_name": ["initiation", "minimum", "ahp"],
                "peak_idx": [np.nan, np.nan, np.nan],
                "peak_value": [np.nan, np.nan, np.nan],
            }
        )

    return pd.DataFrame(
        {
            "peak_name": ["initiation", "minimum", "ahp"],
            "peak_idx": [before_idx, min_idx, after_idx],
            "peak_value": [before_val, min_val, after_val],
        }
    )
