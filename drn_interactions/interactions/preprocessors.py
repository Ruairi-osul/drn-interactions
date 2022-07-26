from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale


class InteractionsPreprocessor:
    def __init__(self, z: bool = True, minmax: bool = False):
        self.z = z
        self.minmax = minmax

    def zscore(self, df):
        if self.z:
            return df.apply(zscore)
        return df

    def min_max(self, df):
        if self.minmax:
            return df.apply(minmax_scale)
        return df

    def __call__(self, df_piv):
        df_piv = self.zscore(df_piv)
        df_piv = self.min_max(df_piv)
        return df_piv
