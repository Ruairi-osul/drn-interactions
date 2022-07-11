from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from drn_interactions.config import Config


def load_derived_generic(name):
    return pd.read_csv(Config.derived_data_dir.absolute() / name)


def load_neurons_derived() -> pd.DataFrame:
    """Load neuron data present after preprocessing

    Returns:
        pd.DataFrame: Neuron data
    """
    df_neuron_props = pd.read_csv(Config.derived_data_dir / "burst_features.csv")
    df_clusters = pd.read_csv(Config.derived_data_dir / "clusters.csv")
    return df_neuron_props.merge(df_clusters)
