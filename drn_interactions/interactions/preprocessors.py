from typing import Optional
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from drn_interactions.transforms.brain_state_spikes import align_spikes_to_states_wide
from scipy.ndimage import gaussian_filter1d


class InteractionsPreprocessor:
    def __init__(
        self,
        z: bool = True,
        minmax: bool = False,
        gaussian_sigma: Optional[float] = None,
        col_thresh: float = 0,
        row_thresh: float = 0,
    ):
        self.z = z
        self.minmax = minmax
        self.gaussian_sigma = gaussian_sigma
        self.col_thresh = col_thresh
        self.row_thresh = row_thresh

    def zscore(self, df):
        if self.z:
            return df.apply(zscore)
        return df

    def min_max(self, df):
        if self.minmax:
            return df.apply(minmax_scale)
        return df

    def exclude_cols(self, df):
        return df.loc[:, (df.sum(axis=0) != self.col_thresh)]

    def exclude_rows(self, df):
        return df.loc[(df.sum(axis=1) != self.row_thresh), :]

    def gaussian_filter(self, df):
        if self.gaussian_sigma is not None:
            return df.apply(lambda x: gaussian_filter1d(x, self.gaussian_sigma))
        return df

    def __call__(self, df_piv):
        df_piv = self.exclude_cols(df_piv)
        df_piv = self.exclude_rows(df_piv)
        df_piv = self.zscore(df_piv)
        df_piv = self.min_max(df_piv)
        df_piv = self.gaussian_filter(df_piv)

        return df_piv.copy()
