## Setup

### Dependencies Installation
To run the tests create and activate conda environment using:

```console
conda env update --file environment.yml --prune
conda activate IntelliLungDataValEnv
```

### Environment File
Running the tests require providing path to the tests config file and the dataset. An example of .env file with MIMIC-IV dataset is given below:

```
VALIDATION_CHECK_CONFIGS_FILE_PATH='configs/validation_checks_mimic.yml'
DATASET_PATH='mimiciv_state_vectors_2024_Dec_4_ai_ready.csv'
```

## Executing Tests

To run the tests cd to current directory (data_validation) and run:
```console
pytest
```
or
```console
python -m pytest
```