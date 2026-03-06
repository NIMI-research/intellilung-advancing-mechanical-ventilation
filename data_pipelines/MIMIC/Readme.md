# Setup instructions

The MIMIC-IV data pre-processing pipeline is run first on the relational database containing MIMIC-IV data and subsequently on the extracted data to get the final state-vector dataset. As such, the pipeline is split into two parts:
- Data extraction
- Data preprocessing

## Pre-requisites & Input
- The data extraction code of the pipeline accesses the MIMIC-IV tables stored in a Postgres database and extracts relevant information using SQL scripts defined in `/sql` 
- After extracting the data from the database and converting it into Pandas dataframes, the data is stored locally as .csv files 
- The second part of the pipeline, data preprocessing, loads these .csv files and runs the data through a series of steps to transform the dataframes into state vectors.

To set up the database locally, refer to the instructions on the website:
https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres

## System requirements
The extracted data size is around `3GB`. Depending on the `NUM_CORES` set for multiprocessing, the memory requirements can vary. For reference, we have used `NUM_CORES=7` with `32GB` memory. Reduce `NUM_CORES` to reduce memory requirements.

## Run instructions

### Run DB query
Set the following in `.env`:

    MIMIC_DBNAME = "your-dbname"
    POSTGRES_USER = "your-dbuser"
    POSTGRES_PASS = "your-dbpass"

This produces the extracted files, with which data extraction can be skipped for the next run.

### Run using extracted files

Either run the query or contact us to get access to the extracted files on our cloud storage.

Set the following in  `.env`:

    INPUT_DATA_PATH="path-to-mimiciv-files"
    OUTPUT_SAVE_PATH="path-to-state-vectors"
    RESOLUTION=3600
    STATE_VECTORS_OUTPUT_NAME="state_vectors_mimiciv"

The pipeline expects the following files to be present:
- `mimiciv_demo.csv`: patient demographics
- `mimiciv_ventilation.csv`: patient invasive ventilation periods
- `mimiciv_inputs.csv`: information about administered fluids and drugs
- `mimiciv_outputs.csv`: information about any fluid output
- `mimiciv.csv`: patient measurement data and lab results
- `actions-rewards-variable-availability.xlsx`: excel file containing information about numerical variables
- `Dict_cath_variables.xlsx`: excel file containing information about categorical variables

## Environment variables

Set variable in `.env.` specific to your hardware and run requirements.

- `NUM_CORES`: according to your system
- `SETUP`: generates necessary .json files containing metadata for relevant variables, but can be skipped if the files already exist

## Execute pipeline

Cd into the current directory for the MIMIC preprocessing pipeline and use the following commands:

Run `python data_extraction_mimic_iv.py`.
Run `python data_preprocessing_mimic_iv.py`

## Output

The extracted data will be saved in the path defined by `INPUT_DATA_PATH` under the folder `mimiciv`, while the pre-processing output will be saved in a timestamped folder in the location defined by `OUTPUT_SAVE_PATH`. The output of the preprocessing pipeline contains:
- `state_vectors_mimiciv.csv`: resulting state vector with `NaN` for variables that patients are completely missing
- `state_vectors_mimiciv_ai_ready.csv`: resulting state vector with only `NaN` free variables

# Pipeline description

1) Loading data
    - ventilation periods
    - patient demographics
    - patient administered fluids/drugs and outputs
    - measurement data and lab results
2) Data cleaning and standardisation
    - encoding string data
    - extracting individual vent episodes
        - merge overlapping intervals
    - filtering patients
    - standardising data, converting units, and removing outliers
3) Vectorise data
    - generate state vectors from available data
    - assign discharge as alive or deceased
4) Quality enhancement
    - compute calculable variables
    - aggregate state vectors by project defined timeframes (60 minutes)
        - calculate dominant categorical variables for each timestep
        - calculate 4h metaparameters
        - adjust episodes start based on first data availability, drop if below minimum duration
        - forward imputation
    - remove ventilation episodes with less than 4h of data and assign episode ids
    - normalise tidal volume
    - handle mode dependent variables
        - Volume-controlled mode: `vent_vt` is an action but `vent_pinsp-peep` is not
        - Pressure-controleld mode: `vent_pinsp-peep` is an action but `vent_vt` is not
    - remove any episodes with less than 4h data and re-assign episode ids
    - calculate `mv_duration`, `pause_until_next`, `post_extubation_interval`
    

