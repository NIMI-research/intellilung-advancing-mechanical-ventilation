# Setup instructions

This data pre-processing pipeline can be run on 1) the eICU database or 2) on extracted files of the database. 

Option 1) runs the pipeline including query and extraction and option 2) directly loads data from extracted files before starting the data processing.

## Pre-requisites & Input
- Option 1: access to a psql database of eICU or 
- Option 2: the extracted files from the queries defined in `sql/`

To set up the database locally, refer to the instructions on the website:
https://eicu-crd.mit.edu/tutorials/install_eicu_locally/

## System requirements
The extracted data size is around `16GB`. Depending on the `NUM_CORES` set for multiprocessing, the memory requirements can vary. For reference, we have used `NUM_CORES=20` with `128GB` memory. Reduce `NUM_CORES` to reduce memory requirements.

## Run instructions

### Run with DB query
Set the following in `.env`:

    QUERY_DB="True"
    DBNAME="your-dbname"
    DBUSER="your-dbuser"
    DBPW="your-dbpw"
    DBOPTIONS="your-dboptions"

This produces the extracted files, with which the step can be skipped for the next run.

### Run using extracted files

Either run the query or contact us to get access to the extracted files on our cloud storage.

Set the following in  `.env`:

    QUERY_DB="False"
    SOURCE_PATH="path-to-eicu-files"

The pipeline expects at least the following files to be present:
- `ventevents.csv`: patient invasive ventilation periods
- `patient.csv`: patient demographics
- csvs of relevant variables (e.g. vitals, ventilation etc.)

## Environment variables

Set variable in `.env.` specific to your hardware and run requirements.

- `NUM_CORES`: according to your system
- `REQUIRED_VARIABLES`: variables needed to be free of `NaN` for the `_ai_ready.csv`. If unspecified, `set_1` variables will be used.

## Execute pipeline

Run `eicu/Main.py`.

## Output

The pre-processing output will be saved in a timestamped folder in the folder `eicu` and contains:
- `state_vectors_eicu.csv`: resulting state vector with `NaN` for variables that patients are completely missing
- `state_vectors_eicu_ai_ready.csv`: resulting state vector with only `NaN` free variables

# Pipeline description

1) Loading data
    - ventilation periods
    - patient demographics
    - csvs of relevant variables (e.g. vitals, ventilation etc.)
2) Preprocessing vent episodes
    - Merge overlapping intervals
    - Filter episodes for minimum duration 
3) Data cleaning
    - Encoding strings
    - Converting units
    - Removing outliers
4) Filtering patients
    - without required demographics or vent mode
    - without set 1 variables
5) Standardize vasopressors
6) Create time windows
    - Create project defined timesteps (60 minutes)
    - Calculate `mv_duration`, `pause_until_next`, `post_extubation_interval`
    - Calculate 4h metaparameters
7) Handling missing data
    - Compute calculable variables
    - Adjust episodes start based on first data availability, drop if below minimum duration
    - Forward imputation
8) Handle mode dependent variables
    - Volume-controlled mode: `vent_vt` is an action but `vent_pinsp-peep` is not
    - Pressure-controleld mode: `vent_pinsp-peep` is an action but `vent_vt` is not
9) Build final state vector
    - Assign episode ids
    

