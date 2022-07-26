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
