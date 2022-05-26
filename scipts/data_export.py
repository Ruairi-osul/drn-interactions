"""
    The data for this project is stored on a local MySQL database. This script connects to the database and exports the data into a set of .parquet files.
"""

from pathlib import Path
from typing import Dict, Optional
from drn_interactions.export import (
    calculate_distance,
    load_events,
    load_spiketimes,
    classify_clusters,
    load_eeg,
)
from drn_interactions.load import get_block_names, get_data_dir
from ephys_queries import db_setup_core
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


def is_footshock(block_name) -> bool:
    return ("shock" in block_name) and ("post" not in block_name)


def save_parquet(
    data_dir: Path, datatype: str, df: pd.DataFrame, block_name: Optional[str] = None
):
    p = data_dir
    if block_name is not None:
        p = p / block_name
    p.mkdir(exist_ok=True)
    fname = str(p / datatype) + ".parquet.gzip"
    df.to_parquet(fname, compression="gzip")


PATH_TO_CLF = Path(r"C:\Users\roryl\repos\ssri_analysis\data\models\cluster_svm.gz")
PATH_TO_SCALER = Path(
    r"C:\Users\roryl\repos\ssri_analysis\data\models\clustering_scaler.gz"
)
PATH_TO_LE = Path(r"C:\Users\roryl\repos\ssri_analysis\data\models\label_encoder.gz")


def main():
    load_dotenv()
    engine, metadata = db_setup_core()
    data_dir = get_data_dir()

    df_neurons = classify_clusters(
        engine,
        metadata,
        PATH_TO_CLF=PATH_TO_CLF,
        PATH_TO_SCALER=PATH_TO_SCALER,
        PATH_TO_LE=PATH_TO_LE,
    )
    df_distance = calculate_distance(engine, metadata)
    save_parquet(data_dir, datatype="neurons", df=df_neurons)
    save_parquet(data_dir, datatype="distances", df=df_distance)
    block_names = get_block_names()
    for block_name in tqdm(block_names):
        if block_name == "pre":
            t_before = 0
        else:
            t_before = 600
        datasets: Dict[str, pd.DataFrame] = {}
        datasets["spiketimes"] = load_spiketimes(
            engine,
            metadata,
            block_name=block_name,
            align_to_block=True,
            t_before=t_before,
        )
        datasets["stft"] = load_eeg(
            engine, metadata, align_to_block=True, t_before=t_before
        )
        if is_footshock(block_name):
            datasets["events"] = load_events(
                engine, metadata, block_name, align_to_block=True
            )
        for data_type, dataframe in datasets.items():
            save_parquet(
                data_dir, block_name=block_name, datatype=data_type, df=dataframe
            )


if __name__ == "__main__":
    main()

