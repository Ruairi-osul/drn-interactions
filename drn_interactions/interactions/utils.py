import pandas as pd
import numpy as np


def zero_diag_df(df):
    X = df.values
    np.fill_diagonal(X, 0)
    return pd.DataFrame(X, index=df.index, columns=df.columns)


def corr_df_to_tidy(
    df,
    index_name="neuron_id",
    value_name="corr",
):
    df = (
        df.copy()
        .reset_index()
        .rename(columns={index_name: "neuron_1"})
        .melt(id_vars="neuron_1", var_name="neuron_2", value_name=value_name)
    )
    return df


def upper_values(X):
    idx = np.triu_indices_from(X, 1)
    return X[idx]


def corrmat_from_vals(vals, n, default_val=0):
    A = np.zeros((n, n)) * default_val
    idx = np.triu_indices(n, 1)
    A[idx] = vals
    A += A.T
    return A


def shuffle_corrmat(A):
    A = A.copy()
    vals = upper_values(A)
    vals_shuffle = np.random.choice(vals, len(vals), replace=True)
    A_shuffle = corrmat_from_vals(vals_shuffle, A.shape[0])
    return A_shuffle


def shuffle_df_corr(df):
    A = df.values
    A_shuffle = shuffle_corrmat(A)
    df_shuffle = pd.DataFrame(A_shuffle, index=df.index, columns=df.columns)
    return df_shuffle
