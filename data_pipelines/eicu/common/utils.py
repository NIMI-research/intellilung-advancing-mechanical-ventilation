import pandas as pd

def get_df_value(df_obj):
    return df_obj.iloc[0] if isinstance(df_obj, pd.Series) else df_obj