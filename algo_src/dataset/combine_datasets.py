import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import dataset.pre_processing_configs as cfg
import dataset.postprocessing as pp

from utils.files import load_yaml


def main(test_dataset_name: str | None = None):
    """Combine datasets (eICU, MIMIC‑IV, HiRID) into a single CSV.

    If *test_dataset_name* is provided (must be one of the dataset names), the
    combined data are additionally split into **train** and **test** parts.

    All artefacts are stored under
    ``logs/datasets/combined_mimic_eicu_hirid/<TIMESTAMP>/`` where
    ``<TIMESTAMP>`` is the current date‑time up to seconds (UTC‑local).
    """

    datasets = ["eicu", "mimiciv_7_action_setup", "hirid"]

    combined_cols: set[str] | None = None
    dataframes: list[pd.DataFrame] = []

    relevant_cols = [
        "episode_id",
        "timepoints",
        "mv_duration",
        "pause_until_next",
        "post_extubation_interval",
        "daemo_discharge",
    ]

    timestep_columns = {
        "eicu": "offset",
        "mimiciv_7_action_setup": "timepoints",
        "hirid": "time_interval",
    }

    for d in datasets:
        # configs = cfg.get_pre_processing_configs(configs_id=d)  # ← keep for future use
        configs = load_yaml(f"configs/{d}/dataset_config.yml")
        data_df = pd.read_csv(configs["dataset_path"])

        # Build intersection of columns present in **all** datasets
        if combined_cols is None:
            combined_cols = set(data_df.columns)
        else:
            combined_cols &= set(data_df.columns)

        id_col = "episode_id"
        timestep_column = timestep_columns[d]

        # Remove rows where vent mode == 0 and harmonise column names
        data_df = pp.remove_vent_mode_zero(data_df, timestep_column).rename(
            columns={id_col: "episode_id", timestep_column: "timepoints"}
        )

        # Keep a flag of origin so we can split train/test later if requested
        data_df["source_dataset"] = d

        dataframes.append(data_df)

    # Final set of columns we will keep
    relevant_cols = list(set(relevant_cols) | combined_cols)

    # Retain only the relevant columns in the same order across datasets
    dataframes = [df[relevant_cols + ["source_dataset"]] for df in dataframes]

    # Combine and reset episode_id consecutively across all datasets
    combined_dataset = pd.concat(dataframes, ignore_index=True)
    combined_dataset["episode_id"] = (
        combined_dataset["episode_id"] != combined_dataset["episode_id"].shift()
    ).cumsum()

    # ---------------------------------------------------------------------
    # Save artefacts
    # ---------------------------------------------------------------------

    timestamp = datetime.now().strftime("%d-%m-%Y")
    out_dir = Path("logs/datasets/combined_mimic_eicu_hirid") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) the full combined dataset
    combined_path = out_dir / f"combined_{'_'.join(datasets)}.csv"
    combined_dataset.to_csv(combined_path, index=False)

    # 2) optional train / test split if a test dataset is specified
    if test_dataset_name:
        if test_dataset_name not in datasets:
            raise ValueError(
                f"Invalid test_dataset_name '{test_dataset_name}'. "
                f"Must be one of: {', '.join(datasets)}."
            )

        test_df = combined_dataset.query("source_dataset == @test_dataset_name")
        train_df = combined_dataset.query("source_dataset != @test_dataset_name")

        test_df.to_csv(out_dir / f"test_{test_dataset_name}.csv", index=False)
        train_df.to_csv(out_dir / f"train_without_{test_dataset_name}.csv", index=False)

    print(f"Datasets successfully written to {out_dir.resolve()}")


if __name__ == "__main__":
    load_dotenv()
    main(test_dataset_name=os.getenv("COMBINE_DATASET_CONFIG_TEST_DATASET"))
