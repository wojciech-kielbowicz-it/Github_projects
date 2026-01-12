import pandas as pd
import numpy as np
import pmdarima as pm


def extrapolate_1999_data(dataframe: pd.DataFrame, indicators_list: list) -> pd.DataFrame:
    """
    Generates data for the year 1999 using linear extrapolation based on data 
    from 2000 and 2001.

    For each indicator provided in `indicators_list`, the function calculates 
    the 1999 value using the linear trend formula:
        val_1999 = 2 * val_2000 - val_2001

    It automatically detects if an indicator in the original dataset contains 
    only non-negative values (e.g., population). In such cases, the extrapolated 
    values for 1999 are clipped to a minimum of 0 to prevent unrealistic negative results.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing indicator data. 
            Must include "terc_code" and "year" columns.
        indicators_list (list): A list of column names (strings) representing 
            the numeric indicators to be extrapolated.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data augmented with 
            the generated rows for 1999, sorted by "terc_code" and "year".
    """
    
    df_out = dataframe.copy()
    
    df_2000 = df_out[df_out["year"] == 2000].set_index("terc_code")
    df_2001 = df_out[df_out["year"] == 2001].set_index("terc_code")
    
    common_counties = df_2000.index.intersection(df_2001.index)

  
    df_1999 = df_2000.loc[common_counties].copy()
    
    df_1999["year"] = 1999
    
    for col in indicators_list:
        if col in df_1999.columns:
            val_2000 = df_2000.loc[common_counties, col]
            val_2001 = df_2001.loc[common_counties, col]
            
            
            df_1999[col] = (2 * val_2000) - val_2001
            
            
            if (dataframe[col] >= 0).all():
                 df_1999[col] = df_1999[col].clip(lower= 0)

    df_1999 = df_1999.reset_index()
    
    df_final = pd.concat([df_out, df_1999], ignore_index= True)
    df_final = df_final.sort_values(by= ["terc_code", "year"]).reset_index(drop= True)
    
    return df_final

import pandas as pd
import numpy as np

def prepare_lagged_features(dataframe: pd.DataFrame, indicators_list: list) -> pd.DataFrame:
    """
    Prepares lagged features by shifting time, calculating deltas, and imputing missing values.

    This function shifts the 'year' column by +1 to create a 1-year lag for all features, 
    calculates 1-year and 5-year differences (deltas) for each indicator grouped by 
    geographical code ('terc_code'), and fills any resulting NaNs with the median 
    value of that group.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing 'terc_code', 'year', 
            and the indicator columns.
        indicators_list (list): List of numerical column names to process.

    Returns:
        pd.DataFrame: A processed DataFrame with shifted years, new delta columns, 
            and imputed missing values.
    """
def prepare_lagged_features(dataframe: pd.DataFrame, indicators_list: list) -> pd.DataFrame:
    """
    Applies a one-year lag to the dataset and calculates annual and five-year indicator deltas.

    This function increments the "year" column by 1 to align socio-economic indicators 
    with the corresponding election year. It computes the 1-year and 5-year differences 
    for each specified indicator, grouped by regional unit (terc_code). Missing values 
    (NaN) are preserved for periods where sufficient historical data is unavailable.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing "terc_code", "year", and indicators.
        indicators_list (list): List of column names for which deltas should be calculated.

    Returns:
        pd.DataFrame: A DataFrame with the shifted timeline and additional columns 
            for 1-year and 5-year deltas.
    """
    df = dataframe.copy()
    
    df["year"] = df["year"] + 1
    df = df.sort_values(by=["terc_code", "year"])
    
    for col in indicators_list:
        df[f"{col}_delta_1_year"] = df.groupby("terc_code")[col].diff(1)
        df[f"{col}_delta_5_years"] = df.groupby("terc_code")[col].diff(5)
        
    return df

def merge_election_data(df_election: pd.DataFrame, df_features: pd.DataFrame, target_years: list) -> pd.DataFrame:
    """
    Filters processed indicator features for specific years and merges them with election data.

    This function extracts rows from the features dataset that match the election 
    cycle years (e.g., every 5 years) and performs an inner join with the election 
    results based on regional codes and the specific year.

    Args:
        df_election (pd.DataFrame): DataFrame containing election results. 
            [cite_start]Expected to have "terc_code" and "year" columns[cite: 8].
        df_features (pd.DataFrame): DataFrame containing socio-economic indicators 
            [cite_start]and their calculated deltas[cite: 33, 35].
        target_years (list): List of integers representing the years to be 
            included in the final dataset (e.g., [2000, 2005, 2010...]).

    Returns:
        pd.DataFrame: A consolidated DataFrame ready for ML modeling, containing 
            both election outcomes and lagged/delta indicators for the chosen years.
    """
    df_feat_filtered = df_features.copy()
    df_el = df_election.copy()
    
    mask = df_feat_filtered["year"].isin(target_years)
    df_feat_filtered = df_feat_filtered[mask]
    
    df_final = pd.merge(
        df_el,
        df_feat_filtered,
        on= ["terc_code", "year", "county"],
        how= "inner",
    )
    
    return df_final


def forecast_arima_to_2030(terc_code: str, dataframe: pd.DataFrame, indicators_list: list, years_list: list) -> list:
    """
    Generates forecasts for specified economic indicators up to the year 2030 using ARIMA models.

    This function isolates data for a single county (identified by `terc_code`), fits an 
    `auto_arima` model for each indicator in `indicators_list` based on historical data, 
    and predicts values for the specified `target_years`.

    Args:
        terc_code (str): The unique TERC identification code for the county.
        dataframe (pd.DataFrame): The input DataFrame containing historical data. 
            Must include "terc_code", "county", "year", and the indicator columns.
        indicators_list (list): A list of strings representing the column names of the 
            economic indicators to be forecasted (e.g., ["gdp_per_capita", "average_gross_salary"]).
        target_years (list): A list of integers representing the future years for which 
            forecasts should be generated.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the forecasted 
            data for a specific year. Each dictionary includes keys for "terc_code", "county", 
            "year", and the predicted values for each indicator in `indicators_list`.
    """
    county_data = dataframe[dataframe["terc_code"] == terc_code].sort_values("year")

    county_name = county_data["county"].iloc[0]
    
    forecasts_by_year = {
        year: {
            "terc_code": terc_code, 
            "county": county_name, 
            "year": year
        } for year in years_list
    }
    
    for col in indicators_list:

        ts = county_data.set_index("year")[col].dropna()

        predictions = []
    

        model = pm.auto_arima(
            ts,
            start_p= 0, 
            start_q= 0,
            max_p= 2, 
            max_q= 2, 
            max_order= 3, 
            d= None, 
            test= "kpss", 
            seasonal= False,         
            stepwise= True, 
            maxiter= 15, 
            method= "nm",  
            error_action= "ignore",
            suppress_warnings= True,
            trace= False, 
            n_jobs= 1
        )
        
        predictions = model.predict(n_periods=len(years_list))
             
        for i, year in enumerate(years_list):
            val = predictions.iloc[i] if hasattr(predictions, "iloc") else predictions[i]
            forecasts_by_year[year][col] = val
            

    return list(forecasts_by_year.values())

