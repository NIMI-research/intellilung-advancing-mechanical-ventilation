## Requirements

* CUDA enabled GPU device (if using CUDA backend for PyTorch)

### Dependencies Installation

For GPU version:
```console
conda env update --file environment.yml --prune
conda activate IntelliLungEnvD33GPU
```


## Getting Started

### Define Config Paths
The pipeline requires defining paths to algorithm and dataset configuration files in a `.env` file. For convenience, examples are provided in the `example.env` file. To set up, follow these steps:
1. Copy configs from `example.env` file to `.env` for the required setup. For example, to run the experiment for the combined dataset discrete actions setup
    ```
   ALGORITHM_CONFIGS_PATH='configs/combined_mimic_eicu_hirid/factored_cql_config.yml'
   DATASET_CONFIG_PATH='configs/combined_mimic_eicu_hirid/dataset_config.yml'
   DIST_FQE_CONFIG_PATH='configs/combined_mimic_eicu_hirid/dist_fqe_config.yml'
   DEVICE=cuda
    ```
2. Make sure dataset is placed at the path specified in the dataset config. For example, dataset configs on path defined above require following dataset file `logs/datasets/combined_mimic_eicu_hirid/combined_eicu_mimiciv_7_action_setup_hirid.csv`. The path to the dataset file can be updated in dataset config.

### Executing End-To-End Pipeline
After having activated conda environment, simply execute following in the terminal
```console
python run_e2e_experiment.py
```

### Policy Coverage

To train the state-conditional density model, run the following command from the `algo_src` directory:

```bash
PYTHONPATH=${PYTHONPATH}:/ python algorithms/eval/hybrid_state_conditional_density.py
```

Before running the command, set the path to the config file in `.env`, for example:

```bash
STATE_CONDITIONAL_DENSITY_CONFIG_PATH=configs/combined_mimic_eicu_hirid_hybrid/state_conditional_density_configs.yml
```

After training, set `AE_EXPERIMENT_PATH` in `.env` to the newly created `hybrid_state_conditional_density` experiment path.

Also set `EXPERIMENT_PATH` in `.env` to the offline RL experiment path for which you want to compute policy coverage.

Then run the following command to evaluate policy coverage:

```bash
PYTHONPATH=${PYTHONPATH}:/ python algorithms/eval/eval_ood_hybrid_ae.py
```
### Hyperparameter Search

To create a grid search experiment, first define the path to the hyperparameter tuning config file.

For example, `algo_src/configs/hyper_param_tune/config.json`:

```json
{
  "root_path": "logs",
  "tune_configs": [
    {
      "number_of_evaluations": 2,
      "base_algo_config_path": "configs/combined_mimic_eicu_hirid_hybrid/hybrid_iql_configs.yml",
      "fqe_config_path": "configs/combined_mimic_eicu_hirid/dist_fqe_config.yml",
      "dataset_config_path": "configs/combined_mimic_eicu_hirid_hybrid/dataset_config.yml",
      "tune": [
        {
          "params": [
            "steps"
          ],
          "values": [
            25000,
            50000,
            100000,
            200000
          ]
        },
        {
          "params": [
            "qf_lr",
            "actor_lr",
            "vf_lr"
          ],
          "values": [
            1e-3,
            1e-4,
            1e-5,
            1e-6
          ]
        }
         .
         .
         .
      ]
    }
  ]
}
```

#### Config fields

- `root_path`: Directory where hyperparameter tuning experiment results will be saved.
- `number_of_evaluations`: Number of times each hyperparameter configuration is repeated.
- `base_algo_config_path`: Path to the base algorithm config file. This should contain the algorithm identifier and default parameters.
- `fqe_config_path`: Path to the FQE evaluator config file. This usually does not need to be changed.
- `dataset_config_path`: Path to the dataset config file. This usually does not need to be changed.
- `tune`: List of parameter grids to search over.

Each entry in `tune` defines one grid dimension:

- `params`: Parameter names to tune.
- `values`: Values to assign.

If multiple parameter names are listed in the same `params` entry, they are updated together using the same value from `values`. For example, `qf_lr`, `actor_lr`, and `vf_lr` will all be assigned the same learning rate for each grid point.

#### Set the config path

In `.env`, set the path to the hyperparameter tuning config file:

```bash
HYPER_PARAM_TUNE_CONFIGS=configs/hyper_param_tune/config.json
```

#### Create experiments

Run the following from the `algo_src` directory:

```bash
PYTHONPATH=${PYTHONPATH}:/ python hyper_param_tune/create_experiments.py
```

This creates a hyperparameter tuning experiment under the configured `root_path`, for example:

```text
logs/hyper-param-tune-2026-03-06 15:47:08.433357
```

#### Run a single experiment

To run one experiment instance (one hyperparameter combination e.g. task 0), use:

```bash
PYTHONPATH=${PYTHONPATH}:/ python -u hyper_param_tune/run_experiment.py \
  --experiment_root_path "logs/hyper-param-tune-2026-03-06 15:47:08.433357" \
  --task_id 0
```

#### Run experiments in parallel on HPC

You can also execute experiments in parallel on an HPC cluster. A template job script is provided in:

```bash
run_hyper_param_tune_jobs.sh
```

#### Reward Tune

Reward tuning follows a similar process to hyperparameter search, except that reward parameters are varied instead of algorithm hyperparameters.

The config file is defined in, for example:

```bash
configs/reward_tune/config.json
```

The scripts for creating and running reward tuning experiments are located in the `reward_tune` directory.

To run experiments in parallel on an HPC cluster, a template job script is provided:

```bash
run_reward_tune_jobs.sh
```

### Wandb
The experiment performance tracking is done using wandb. To disable wandb for any script, add `WANDB_DISABLED=true` as argument before the run command e.g. the end-to-end script above can be executed without wandb using
```console
WANDB_DISABLED=true python run_e2e_experiment.py
```



