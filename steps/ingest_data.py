import logging

import pandas as pd
from typing_extensions import (
    Annotated,  # or `from typing import Annotated on Python 3.9+
)
from zenml import step

from steps.source.data_loader import DataLoader


@step(enable_cache=False)
def ingest_data(
    table_name: str ,
    for_predict: bool = False,
) -> pd.DataFrame:
    """Reads data from sql database and return a pandas dataframe.

    Args:
        data: pd.DataFrame
    """
    try:
        data_loader = DataLoader("postgresql://postgres:sunbreathing@localhost:5433/cs001test2")
        data_loader.load_data(table_name)
        df = data_loader.get_data()
        if for_predict:
            df.drop(columns=["unit_price"], inplace=True)
        print(df.head())
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error(e)
        raise e