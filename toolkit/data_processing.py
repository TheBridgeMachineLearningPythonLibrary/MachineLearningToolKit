import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List


def extract_date(df, date_column_name):
    '''
    Function to extract the data of a date, from a column that does not have date format.
    The function also creates a column for date and year.

    Parameters
    ----------
    df : dataframe
        It is the dataset where the column that we have to extract the data is located.
    date_column_name : str
        is the column that has data in string. From here the data is extracted and converted to date format.

    
    Return
    ----------
        df : Dataframe with the changes made

    '''
    match = re.search(r'\d{4}.\d{2}.\d{2}', date_column_name)
    date = datetime.strptime(match.group(), '%Y-%m-%d').date()

    df['Year'] = df[date_column_name].dt.year
    df['Month'] = df[date_column_name].dt.month
    
    return df

def list_categorical_columns(df):
    '''
    Function that returns a list with the names of the categorical columns of a dataframe.

    Parameters
    ----------
    df : dataframe
    
    Return
    ----------
        features: list of names

    '''
    features = []

    for c in df.columns:
        t = str(df[c].dtype)
        if "object"  in t:
            features.append(c)
    return features



def uniq_value(list_values:list):
    '''
    Function returning the unique values from a list.
    Parameters
    ----------
    list_values:list
    Return
    ----------
    unique: list of unique values
    '''
    unique = []
    for i in list_values:
        if i not in unique:
            unique.extend(list_values)
    return unique

def last_columndf(df,feature):
    '''
    Function will return the dataframe with the column entered at the last position
    ----------
    df: dataframe
    feature: to move at the last position

    Return
    ----------
    df: dataframe
    '''

    lista=list(df.columns.values)
    lista.remove(feature)
    lista.append(feature)
    df=df[lista]
    return df



def log_transform_data(df: pd.DataFrame, col_ignore: List[str]) -> pd.DataFrame:
    '''
    Log transform the numeric columns and recombine back with the non-numeric
    Option to skip specific numeric columns

    Parameters
    ----------
    df : pd.DataFrame
        The dataset we want to carry out log transform on
    col_ignore : List[str]
        A list of numeric column names that we want to skip the log transform for

    
    Return
    ----------
    pd.DataFrame with same dimensions as input
    '''
    df_ignore = df[col_ignore]
    df_rest = df.drop(columns = col_ignore)
    numeric = df_rest.select_dtypes(include=np.number).apply(np.log1p)
    non_numeric = df_rest.select_dtypes(exclude=np.number)
    
    return pd.concat([numeric, non_numeric, df_ignore], axis = 1)