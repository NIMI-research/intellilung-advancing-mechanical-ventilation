# Setup instructions for HiRID's Data Preprocessing Pipeline:

This data pre-processing pipeline needs to be run on extracted files of the HiRID's database.

## Pre-requisites & Input

To set up the database locally, refer to the instructions on the website to download files from HiRID's dataset at [Physionet](https://physionet.org/content/hirid/1.1.1/#files-panel)

1. To extract 5-min intervals source tables:
- Run the _cluster_ .py files from `dm&stats` folder. `dm_merged`, `dm_fluids`, `dm_vasoactive` and `dm_ventilation` output's parquet files contain different sets of variables for ICU patients in 5-min intervals. 
- Save these parquet files in specific subfolders with the previous names within `source_tables`.

2. To load demographic data:
- Place the `General Table Schema` file from the downloaded HiRID source files into the `data_pipelines/HiRID/source_tables/static_table` directory, and rename it to _general_table.csv_.

## System requirements
Depending on the `NUM_CORES` set for multiprocessing, the memory requirements can vary. For reference, we have used `NUM_CORES=7` with `16GB` memory. Change `NUM_CORES` to reduce or increase memory requirements.

## Environment variables
Make a copy of _.env.example_ and save it as _.env_. Specify settings according to your setup:
- `NUM_CORES`: according to your system
- `SOURCE_PATH`: full path to the HiRID's pipeline folder (ending with _data_pipelines/HiRID_)
- `OUTPUT_SAVE_PATH`: location where all outputs from the data extraction and preprocessing pipeline will be stored.
- The rest `.env` variables are names for specific generated files.

## Execute pipeline

Run the _main.ipynb_ or _main.py_ file at HiRID's general folder.

Set `HiRID/common/config/PARQUET_TEST_NUM` to 0 to run the pipeline on all files, or to 1 to test the code with the minimum number of source files.

## Output

The pre-processing output will be saved in the `OUTPUT_SAVE_PATH` folder within the HiRID folder, containing a set of intermediate tables.

The final table is named `hirid_state_vector_60min_setfullai_ready`.

# Pipeline description

We defined 'patient cohort' as the patients fulfilling the next criteria:
- Patients with invasive mechanical ventilation (IMV).
- episode duration higher or equal than 4h.
- Patients with valid and in-range values for height and weight.

The pipeline goes through a series of stages.

1) Data Filtering:
	- Load and filter static/demographic table. 
	- Extraction of IMV episodes (merged with <6h time gap between them) and ventilated data. 
	- Selection of patient cohort. 
	- Output tables:
		* vent_df_reduced: ventilation data in 5-min intervals for patient cohort, including time gaps without IMV.
		* vent_episodes: >=4h IMV grouped episodes for patient cohort.
		* cohort_patients_list: list of patient cohort.

2) Data Loading
	- Joining of dm source tables by patient cohort. 
	- Creation of new variables. 
	- Merging source tables with demographic and ventilation data. 
	- Output tables:
		* state_vector_raw: raw data in 5-min intervals on patient cohort, before applying unit conversion and outlier removal, and any other type of data transformation whatsoever. It contains only >=4h IMV data from patient cohort.
		* demog_table: demographic data containing static variables such as sex, weight, ideal body weight, height, discharge, etc

3) Data Cleaning & Standardization
	- Conversion of variable units.  
	- Converting outliers to NA. 
	- Transformation and re-encoding of variables. 
	- Output table: 
		* state_vector_not_imputed: dataframe in 5-min intervals after unit conversion, outlier removal and data transformation. It contains rows of patient cohort for episodes fulfilling inclusion ventilation criteria.
	- Forward propagation of non-null values for each patient and ventilation episode (mv_id).
	- Output table: 
		* state_vector_imputed: dataframe in 5-min intervals after forward propagation of variables by each patient and episode of mechanical ventilation.

4) Time-windowing Creation
	- Creation of time-windowed state vector according to resolution.
	- Recalculate composite variables.
	- Output table: 
		* state_vector_time_windows: dataframe in resolution intervals after applying time windowing.

5) Cutting Point & Selection of set variables
	- Selection of specific set variables of interest for outputing final state vector (set_variables).
	- Implementation of a cutoff point starting from windows within episodes where all specified variables have valid values.
	- Selection of episodes with a duration of >=4h, as the episode duration may be shorter after applying the cutoff point.
	- Output table: 
		* hirid_state_vector_60min_setfullai_ready: time windowed state vector after cut off and selecting >=4h episodes, for model training. 

5) Quality control
	- Check the number of rows in the final table.
	- Export the final output to CSV format.
	- Instructions for data validation are provided. Please see `data_pipelines/data_validation` README file.

For more information on specific tables and functions, individual .py files at the `HiRID/data_preprocessing` directory contain additional detailed descriptions.