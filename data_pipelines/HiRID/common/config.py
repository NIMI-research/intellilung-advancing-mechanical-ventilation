from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    source_path: str
    output_save_path: str
    num_of_cores: int
    path_to_source_tables: str
    static_table: str
    state_vector_cohort: str
    state_vector_raw_name: str
    state_vector_not_imputed_name: str
    state_vector_imputed_name: str
    state_vector_time_windows_name: str
    min_duration: int 
    resolution: int

    # PARQUET_TEST_NUM is used to define how many parquet files from each subfolder (dm, fluid, vasoactive, ventilation) 
    # the pipeline works on. Setting this parameter to 0 will instead run the pipeline for ALL parquet files
    PARQUET_TEST_NUM: int = 0
    # Create a time window state vector departing from the non-imputed dataset. Yes (1) or No (0)
    create_nonimputed: int = 1
    # To change specific set of variables of interest, modify set_variables and set_variables_name in main file

    # Read previously loaded state_vector_imputed for running Data Vectorization chunk in main.py if set to 1
    load_imputed: int = 0
    # Read previously loaded state_vector_nonimputed for running Data Vectorization chunk in main.py if set to 1
    load_nonimputed: int = 0
    # Read previously loaded state_vector_time_windows_imputed for running set selection in main.py if set to 1
    load_timewindows: int = 0

    class Config:
        env_file = ".env"


settings = Settings()


