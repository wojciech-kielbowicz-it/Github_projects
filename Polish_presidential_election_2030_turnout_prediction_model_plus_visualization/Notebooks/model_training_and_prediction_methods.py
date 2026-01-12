import pandas as pd
import numpy as np

# Methods: 

def prepare_train_and_test_data(model_set_dataframe: pd.DataFrame, target_column: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing sets based on a specific year.

    The test set is created using data from the specified `year`. The training set 
    is derived from all other years, with rows containing missing values (NaN) 
    in the `target_column` removed.

    Args:
        model_set_dataframe (pd.DataFrame): The source dataframe containing the complete dataset.
        target_column (str): The name of the target variable column used to drop NaN rows 
            in the training set.
        year (int): The specific year used to segregate the test data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two dataframes:
            - train_dataframe: The dataset for training (all years except `year`, cleaned of target NaNs).
            - test_dataframe: The dataset for testing (only data from `year`).
    """
    test_dataframe = model_set_dataframe[model_set_dataframe["year"] == year].copy()

    train_dataframe = model_set_dataframe[model_set_dataframe["year"] != year].copy()
    train_dataframe = train_dataframe.dropna(subset=[target_column])

    return train_dataframe, test_dataframe