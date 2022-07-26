from .cross_correlations import ccf
from drn_interactions.transforms.spikes import pop_population_train
import pandas as pd
import numpy as np


class PopulationCoupling:
    def __init__(self, nlags=50, method="ccf", min_spikes=0):
        self.nlags = nlags
        self.method = method
        self.min_spikes = min_spikes

    def fit(self, df_piv):
        """Get population couplings for each column in a dataframe

        Args:
            df_piv (pd.DataFrame): Index is time bin, columns are neurons, values are activity

        Returns:
            pd.DataFrame: A
        """
        time_interval = np.diff(df_piv.index.values)[0]
        frames = []
        df_piv = self.filter_active(df_piv, thresh=self.min_spikes)
        for neuron_id in df_piv.columns:
            neuron, pop_train = pop_population_train(df_piv, neuron_id)
            if neuron[neuron_id].sum() == 0:
                continue
            lags, vals = ccf(neuron[neuron_id], pop_train["population"], adjusted=False)
            time = lags * time_interval
            zero_index = int(np.argwhere(time == 0))
            start_idx = zero_index - self.nlags
            stop_idx = zero_index + self.nlags
            frames.append(
                pd.DataFrame(
                    {"neuron_id": neuron_id, "lag": lags, "time": time, "cc": vals}
                ).iloc[start_idx:stop_idx]
            )
        self.couplings_ = pd.concat(frames)
        return self

    def get_couplings(self, method="zerolag"):
        """Get couplings

        Args:
            method (str, optional): One of {'zerolag', 'all'}. Defaults to "zerolag".

        Raises:
            ValueError: If unknown method

        Returns:
            pd.DataFrame: Dataframe of couplings
        """
        df = self.couplings_
        if method == "zerolag":
            return df.query("lag == 0")
        elif method == "all":
            return df
        else:
            raise ValueError("Unknown method")

    @staticmethod
    def filter_active(df, thresh=0):
        sums = df.sum()
        to_exclude = sums[sums <= thresh].index.values
        return df[[c for c in df.columns if c not in to_exclude]]

    def __call__(self, df_piv, method="zerolag"):
        self.fit(df_piv)
        return self.get_couplings(method)


def popcup_zerolag(df, nlags=100, name="popcup"):
    out = PopulationCoupling(nlags=nlags).get_couplings(df)
    return out.loc[lambda x: x.lag == 0].rename(columns={"cc": name})[
        ["neuron_id", name]
    ]
