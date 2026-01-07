import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pmdarima as pm
from joblib import Parallel, delayed
from functools import reduce

# Methods:

def get_row_by_county_as_df(dataframe: pd.DataFrame, county_name: str) -> pd.DataFrame:
    """
    Filters the DataFrame for a specific county and returns only voting-related columns.

    Args:
        dataframe (pd.DataFrame): The main DataFrame.
        county_name (str): The name of the county to filter by.

    Returns:
        pd.DataFrame: A subset containing 'authorized_voters' and 'votes_cast' for the county.

    First use -> Presidential election 2000 data 
    """
    return  (
        dataframe.loc[
            dataframe["county"] == county_name, [
                "authorized_voters", 
                "votes_cast"
            ]
        ]
    )

def get_new_row_as_dict(county_name: str, selected_row: pd.DataFrame, percentage: float) -> dict:
    """
    Creates a dictionary representing a new data row with calculated vote counts.
    
    WARNING: This function relies on a global variable 'selected_row' 
    which must be defined in the scope before calling this function.

    Args:
        county_name (str): The name of the county for the new row.
        percentage (float): The multiplier to apply (e.g., 0.5 for 50%).

    Returns:
        dict: A dictionary with the county name and calculated integer vote values.

    First use -> Presidential election 2000 data  
    """
    return {
        "county": county_name,
        "authorized_voters": int(selected_row.iloc[0, 0] * percentage), 
        "votes_cast": int(selected_row.iloc[0, 1] * percentage)
    }

def update_terc_codes(target_df: pd.DataFrame, source_df: pd.DataFrame, join_column: str = "county", code_column: str = "terc_code") -> pd.DataFrame:
    """
    Updates the TERC code column in the target DataFrame using values from the source DataFrame, 
    ensuring the final column is of string type and preserves leading zeros.

    The function performs a left join, resolves duplicates by prioritizing codes that match 
    the first two characters of the original code, and ensures the output TERC codes 
    are stored as strings (preventing the loss of leading zeros like '02').

    Args:
        target_df (pd.DataFrame): The main DataFrame (e.g., presidential election data).
        source_df (pd.DataFrame): The reference DataFrame containing TERC codes.
        join_column (str): The column name to merge on (default: "county").
        code_column (str): The name of the TERC code column (default: "terc_code").

    Returns:
        pd.DataFrame: A new DataFrame with updated TERC codes.

    First use -> Presidential election 2000 data
    """
    target_df = target_df.copy()
    target_df["_original_index"] = target_df.index

    source_view = source_df[[join_column, code_column]].copy()
    source_view[code_column] = source_view[code_column].astype(str)

    merged = target_df.merge(
        source_view, 
        on=join_column, 
        how="left", 
        suffixes=("", "_new")
    )

    old_prefix = merged[code_column].astype(str).str.slice(0, 2)
    new_prefix = merged[f"{code_column}_new"].astype(str).str.slice(0, 2)
    merged["match_score"] = (old_prefix == new_prefix).astype(int)

    merged.sort_values(
        by=["_original_index", "match_score"], 
        ascending=[True, False], 
        inplace=True
    )
    merged.drop_duplicates(subset= ["_original_index"], keep= "first", inplace= True)
    merged[code_column] = merged[f"{code_column}_new"]

    merged.set_index("_original_index", inplace= True)
    merged.index.name = None
    merged.sort_index(inplace= True)
    merged = merged.drop(columns= [f"{code_column}_new", "match_score"])

    return merged

def extrapolate_backwards_one_county(dataframe: pd.DataFrame, column_name: str, county_name: str, year_x: int, year_y: int) -> None:
    """
    Performs backward extrapolation for a specific column restricted to a single county.

    This function calculates values for an earlier time period based on the data 
    associated with the provided years, modifying the DataFrame in-place.

    Args:
        dataframe (pd.DataFrame): The dataset containing the values to extrapolate.
        column_name (str): The name of the column to update or calculate.
        county_name (str): The name of the county to filter the operation by.
        year_x (int): The first year point used in the extrapolation logic.
        year_y (int): The second year point used in the extrapolation logic.

    First use -> Popualtion density data
    """
    target_years = list(range(year_x, year_y + 1))

    mask = (dataframe["county"] == county_name)
    sub_dataframe = dataframe[mask]

    mask = (
        (sub_dataframe["year"] >= year_y + 1) 
        & 
        (sub_dataframe["year"] <= 2024)
    )

    train_data = sub_dataframe[mask].sort_values("year")
        
    X = train_data["year"].values
    y = train_data[column_name].values
            
    slope, intercept = np.polyfit(X, y, 1)
    predict = lambda yr: slope * yr + intercept
            
    for year in target_years:

        mask = (
            (dataframe["county"] == county_name)
            &
            (dataframe["year"] == year)
        )
        
        forecast = predict(year)
        dataframe.loc[mask, column_name] = round(forecast, 1)

def extrapolate_missing_2000_2001(dataframe: pd.DataFrame, column_name: str) -> None:
    """
    Extrapolates missing values for the years 2000 and 2001 based on a linear trend
    derived from 2002-2004 data, applying changes in-place per county.

    For each unique county, the function fits a linear regression model (degree 1)
    using data from the years 2002, 2003, and 2004. It then predicts values for
    2000 and 2001. If a predicted value is non-positive, the value from 2002
    ('population_density') is used as a fallback.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing columns 'county',
            'year', 'population_density', and the target column to extrapolate.
        column_name (str): The name of the column to populate with extrapolated values.
    
    First use -> Population density data
    """

    counties = dataframe["county"].unique()
    
    for county in counties:
        mask = dataframe["county"] == county
        sub_dataframe = dataframe[mask]

        train_data = sub_dataframe[sub_dataframe["year"].isin([2002, 2003, 2004])].sort_values("year")
        val_2002 = sub_dataframe[sub_dataframe['year'] == 2002][column_name].values[0]
            
        X = train_data["year"].values
        y = train_data[column_name].values
        
        slope, intercept = np.polyfit(X, y, 1)
        predict = lambda year: slope * year + intercept
        
        mask = (
            (dataframe["county"] == county) 
            & 
            (dataframe["year"] == 2001)
        )

        if mask.any():
            forecast = predict(2001)
            if forecast <= 0:
                dataframe.loc[mask, column_name] = round(val_2002, 1)
            else:
                dataframe.loc[mask, column_name] = round(forecast, 1)

        mask = (
            (dataframe["county"] == county) 
            & 
            (dataframe["year"] == 2000)
        )
                         
        if mask.any():
            forecast = predict(2000)
            if forecast <= 0:
                dataframe.loc[mask, column_name] = round(val_2002, 1)
            else:
                dataframe.loc[mask, column_name] = round(forecast, 1)

def backcasting_arima(county_id: str, values: np.ndarray, periods: int) -> tuple:
    """
    Performs backcasting using an ARIMA model to estimate historical values for a specific county.

    The function reverses the provided time series data to treat the past as the future,
    fits an auto_arima model, and generates predictions for the specified number of periods.
    It handles short time series (less than 3 observations) and potential model fitting exceptions
    by returning None.

    Args:
        county_id (str): The unique identifier for the county.
        values (np.ndarray): A numpy array of observed values, sorted chronologically.
        periods (int): The number of historical periods (years) to backcast.

    Returns:
        tuple | None: A tuple containing (county_id, predicted_values) where predicted_values
        are rounded to the nearest integer, or None if the input series is too short
        or model fitting fails.

    First use -> Population 70 plus data
    """

    if len(values) < 3: 
        return None

    try:
        training_data = values[::-1]

        model = pm.auto_arima(
            training_data, 
            start_p= 0, 
            start_q= 0,
            max_p= 3, 
            max_q= 3,
            m= 1,              
            d= None,           
            seasonal= False,   
            stepwise= True,    
            suppress_warnings= True,
            error_action= "ignore",
            maxiter= 15,       
            n_jobs= 1          
        )

        prediction = model.predict(n_periods= periods)
        
        return (county_id, prediction.round(0))
        
    except Exception:
        return None

def update_df_after_arima_backcasting(dataframe: pd.DataFrame, valid_results: tuple, column_name: str, nearest_year: int) -> pd.DataFrame:
    """
    Updates the target DataFrame with backcasted values derived from ARIMA predictions.

    This function iterates through the prediction results, maps them to the correct
    historical years starting backwards from `nearest_year`, and performs an update operation
    on the main DataFrame based on a composite index of 'county' and 'year'.

    Args:
        dataframe (pd.DataFrame): The target DataFrame containing 'county' and 'year' columns.
        valid_results (tuple): A collection of tuples (county_id, predictions) returned by the backcasting function.
        column_name (str): The name of the column where the backcasted values should be inserted.
        nearest_year (int): The most recent historical year to fill (e.g., if data exists from 2002, this might be 2001).

    Returns:
        pd.DataFrame: The DataFrame with updated values in the specified column and a reset index.

    First use -> Population 70 plus data
    """
    counties = []
    years = []
    values = []

    for county, preds in valid_results:
        for i, val in enumerate(preds):
            counties.append(county)
            years.append(nearest_year - i)
            values.append(val)
        
    updates_df = pd.DataFrame({
        "county": counties,
        "year": years,
        column_name: values
    })

    dataframe = dataframe.set_index(["county", "year"])
    updates_df = updates_df.set_index(["county", "year"])

    dataframe.update(updates_df)
    
    dataframe = (
        dataframe
        .reset_index()
        .astype({"year": "int64"})
    )

    return dataframe

def merge_df_by_voivodeship(main_df: pd.DataFrame, sec_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """
    Merges a county-level DataFrame with a voivodeship-level DataFrame based on 
    the year and the voivodeship code extracted from the TERC code.

    The function creates a common merge key by taking the first two characters 
    of the 'terc_code' column in both DataFrames. This allows for broadcasting 
    a voivodeship-level value to every county located within that voivodeship.

    Args:
        main_df (pd.DataFrame): The main DataFrame containing county-level data 
            (expecting 4-digit TERC codes).
        new_df (pd.DataFrame): The DataFrame containing voivodeship-level data.
        value_column (str): The name of the column in `new_df` to be merged 
            into `main_df`.

    Returns:
        pd.DataFrame: A new DataFrame containing all columns from `main_df` 
        plus the `value_column` matched by voivodeship and year.

    First use -> Merging indicators data
     
    """
    new_subset = sec_df[["year", "terc_code", value_column]].copy()
    
    new_subset["join_key"] = (
        new_subset["terc_code"]
        .astype(str)
        .str.zfill(2)
        .str[:2]
    )
    
    new_subset = new_subset[["year", "join_key", value_column]]

    main_df["temp_voiv_key"] = (
        main_df["terc_code"]
        .astype(str)
        .str.zfill(4)
        .str[:2]
    )

    merged_df = pd.merge(
        main_df,
        new_subset,
        left_on= ["year", "temp_voiv_key"], 
        right_on= ["year", "join_key"],     
        how= "left"                         
    )

    merged_df = merged_df.drop(columns= ["temp_voiv_key", "join_key"])
    
    return merged_df
