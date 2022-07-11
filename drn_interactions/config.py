from pathlib import Path
import os


def _get_base_dir(project_dirname: str) -> Path:
    paths = [
        p
        for p in Path(__file__).absolute().parents
        if p.name.lower() == project_dirname.lower()
    ]
    if len(paths) == 0:
        basepath = Path(os.getcwd())
    else:
        basepath = paths[0]
    return basepath


def get_default_data_dir(project_dirname="DRN Interactions") -> Path:
    basepath = _get_base_dir(project_dirname)
    return basepath / "data"


def get_default_fig_dir(project_dirname="DRN Interactions") -> Path:
    basepath = _get_base_dir(project_dirname)
    return basepath / "figs"


def get_default_derived_data_dir(project_dirname="DRN Interactions") -> Path:
    return get_default_data_dir(project_dirname=project_dirname) / "derived"


class Config:
    data_dir = get_default_data_dir("DRN Interactions")
    derived_data_dir = get_default_data_dir("DRN Interactions")
    fig_dir = get_default_fig_dir("DRN Interactions")

    eeg_states = ("sw", "act")
    cluster_col = "wf_3"


class ExperimentInfo:
    block_names = (
        "pre",
        "base_shock",
        "post_base_shock",
        "chal",
        "chal_shock",
        "post_chal_shock",
        "way",
        "pre",
        "chal",
        "way",
    )
    group_names = (
        "acute_citalopram",
        "acute_saline",
        "shock",
        "sham",
        "acute_cit",
        "acute_sal",
    )
    cit_groups = ("acute_citalopram", "acute_cit")
    sal_groups = ("acute_saline", "acute_sal")
    foot_shock_sessions_all = (
        "hamilton_10",
        "hamilton_03",
        "hamilton_04",
        "hamilton_09",
        "hamilton_31",
        "hamilton_38",
        "hamilton_37",
        "hamilton_35",
        "hamilton_36",
        "hamilton_32",
    )
    foot_shock_sessions_10min = (
        "hamilton_10",
        "hamilton_03",
        "hamilton_04",
        "hamilton_09",
        "hamilton_31",
        "hamilton_38",
        "hamilton_37",
        "hamilton_35",
        "hamilton_36",
        "hamilton_32",
    )
    eeg_sessions = (
        "ESHOCK_01",
        "ESHOCK_02",
        "ESHOCK_03_LOC1",
        "ESHOCK_04_LOC1",
        "ESHOCK_06_LOC1",
        "ESHOCK_07_LOC1",
        "ESHOCK_08_LOC1",
        "ESHOCK_09_LOC1",
        "acute_11",
        "acute_12",
        "acute_14",
        "acute_15",
        "acute_16",
        "hamilton_03",
        "hamilton_04",
        "hamilton_09",
        "hamilton_10",
    )
    chal1_sessions = (
        "hamilton_10",
        "hamilton_03",
        "hamilton_04",
        "hamilton_09",
        "hamilton_31",
        "hamilton_38",
        "hamilton_37",
        "hamilton_36",
        "hamilton_32",
        "acute_15",
        "acute_16",
        "acute_01",
        "acute_12",
        "acute_11",
    )
    way_sessions = (
        "hamilton_09",
        "hamilton_31",
        "hamilton_38",
        "hamilton_37",
        "hamilton_36",
        "hamilton_32",
        "acute_15",
        "acute_01",
        "acute_12",
        "acute_11",
    )
