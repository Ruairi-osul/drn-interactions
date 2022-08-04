import seqnmf
import numpy as np


class SeqNMF:
    def __init__(self, K, L, Lambda, W_fixed=False, allow_shift=True):
        self.K = K
        self.L = L
        self.Lambda = Lambda
        self.W_fixed = W_fixed
        self.allow_shift = allow_shift

    def fit(self, X, y=None):
        W, H, cost, loadings, power = seqnmf.seqnmf(
            X,
            K=self.K,
            L=self.L,
            Lambda=self.Lambda,
            W_fixed=self.W_fixed,
            shift=self.allow_shift,
        )

        self.W_ = W
        self.H_ = H
        self.cost_ = cost
        self.loadings_ = loadings
        self.power_ = power
        return self

    def sequenciness_score(self, X, y=None):
        X_obs = np.array(X)
        X_time_shuffled = X_obs[:, np.random.permutation(X.shape[1])]
        X_neuron_shuffled = np.apply_along_axis(np.random.permutation, 1, X_obs)
        _, _, _, _, power_obs = seqnmf.seqnmf(
            X_obs,
            K=self.K,
            L=self.L,
            Lambda=self.Lambda,
            W_fixed=self.W_fixed,
            shift=self.allow_shift,
        )
        _, _, _, _, power_time_shuffled = seqnmf.seqnmf(
            X_time_shuffled,
            K=self.K,
            L=self.L,
            Lambda=self.Lambda,
            W_fixed=self.W_fixed,
            shift=self.allow_shift,
        )
        _, _, _, _, power_neuron_shuffled = seqnmf.seqnmf(
            X_neuron_shuffled,
            K=self.K,
            L=self.L,
            Lambda=self.Lambda,
            W_fixed=self.W_fixed,
            shift=self.allow_shift,
        )
        sequenciness_score = (power_obs - power_time_shuffled) / (
            power_obs / power_neuron_shuffled
        )
        return sequenciness_score
