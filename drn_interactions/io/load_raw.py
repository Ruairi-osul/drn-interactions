import pandas as pd
from drn_interactions.config import Config


def load_recordings() -> pd.DataFrame:
    """Load neurons data from the data dir to a DataFrame

    Returns:
        pd.DataFrame: DataFrame of neurons data
    """
    p = Config.data_dir / "recordings.parquet.gzip"
    return pd.read_parquet(p)


def load_neurons() -> pd.DataFrame:
    """Load neurons data from the data dir to a DataFrame

    Returns:
        pd.DataFrame: DataFrame of neurons data
    """
    p = Config.data_dir / "neurons.parquet.gzip"
    return pd.read_parquet(p)


def load_distances() -> pd.DataFrame:
    """Load distance data

    Returns:
        pd.DataFrame: DataFrame with one row per combination of neurons and the distance between them
    """
    p = Config.data_dir / "distances.parquet.gzip"
    return pd.read_parquet(p)


def load_spikes(block_name: str) -> pd.DataFrame:
    """Load spiketimes data from an experimental block

    Args:
        block_name (str): Name of the block [see get_block_names]

    Returns:
        pd.DataFrame: A pandas DataFrame with one row per spiketime
    """
    p = Config.data_dir / block_name / "spiketimes.parquet.gzip"
    return pd.read_parquet(p)


def load_eeg(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "eeg_stft.parquet.gzip"
    return pd.read_parquet(p)


def load_eeg_raw(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "eeg_raw.parquet.gzip"
    return pd.read_parquet(p)


def load_eeg_ts(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "eeg_band_ts.parquet.gzip"
    return pd.read_parquet(p)


def load_lfp(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "lfp_stft.parquet.gzip"
    return pd.read_parquet(p)


def load_lfp_raw(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "lfp_raw.parquet.gzip"
    return pd.read_parquet(p)


def load_lfp_ts(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "lfp_band_ts.parquet.gzip"
    return pd.read_parquet(p)


def load_events(block_name: str) -> pd.DataFrame:
    p = Config.data_dir / block_name / "events.parquet.gzip"
    return pd.read_parquet(p)


def load_waveforms() -> pd.DataFrame:
    p = Config.data_dir / "waveforms.parquet.gzip"
    return pd.read_parquet(p)
