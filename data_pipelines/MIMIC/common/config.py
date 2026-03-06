import os

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    input_data_path: str
    output_save_path: str
    num_of_cores: int
    mimic_dbname: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_pass: Optional[str | int] = None
    host: Optional[str] = None
    postgres_port: Optional[int] = None
    path_to_sql_queries: Optional[str] = None
    state_vectors_output_name: Optional[str] = None
    resolution: int
    use_postgres: Optional[bool] = False
    load_from_csv: Optional[bool] = False

    # SETUP is used to re-generate knowledge base dictionaries in case some variables/itemIds are changed
    # SAVE_TO_CSV is used to enable querying the PostgreSQL database and then to save the results to csv files.
    # It should be mentioned that this will only be done to "labevents" and "chartevents" because the other
    # tables are far smaller and don't require saving to a csv to save time
    setup: bool = True

    # PATIENT_TEST_NUM is used to define how many stay ids should the pipeline work on. Setting this parameter to
    # 0 or a negative number will instead run the pipeline for ALL stay ids
    stay_id_test_num: int = -1

    class Config:
        env_file = ".env"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute the path once during initialization
        self.path_to_sql_queries = os.path.join(os.getcwd(), "sql")


settings = Settings()
