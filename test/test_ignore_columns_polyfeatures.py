import pandas as pd
from typing import List
from sklearn.preprocessing import PolynomialFeatures

from toolkit.machine_learning import ignore_columns_polyfeatures


def test_log_transform_data_ignore():
    df = pd.DataFrame({'a':[0, 0, 1, 0],
                  'b': [16, 7, 6, 16],
                  'c':[61, 57, 16, 36],
                  'd':['12','22','13','44'],
                  'e':['Green','Red','Blue','Yellow'],
                  'f':[1, 11, 23, 66]})

    df_processed = ignore_columns_polyfeatures(df, variables_to_ignore = ['a', 'd', 'e'], n = 2)

    pd.testing.assert_frame_equal(df[['a', 'd', 'e']], df_processed[['a', 'd', 'e']])



def test_log_transform_data_ignore():
    df = pd.DataFrame({'a':[0, 0, 1, 0],
                  'b': [16, 7, 6, 16],
                  'c':[61, 57, 16, 36],
                  'd':['12','22','13','44'],
                  'e':['Green','Red','Blue','Yellow'],
                  'f':[1, 11, 23, 66]})

    df_processed = ignore_columns_polyfeatures(df, variables_to_ignore = ['a', 'd', 'e'], n = 2)

    assert len(df_processed.columns) == 13

















def ignore_columns_polyfeatures(X: pd.DataFrame, variables_to_ignore: List[str], n: int) -> pd.DataFrame:
    '''
    This function takes a dataframe as input and will create n polynomial features for all columns except those specified to ignore
    It is intended to be used to ignore binary columns for example and to be included in a Pipeline

    Parameters
    ----------
    X : dataframe
        It is the dataset we want to selectively create polynomial features
    variables_to_ignore : List[str]
        a list of column names to ignore in the polynomial feature creation
    n : int
        the degree for the polynomial fearture creation

    
    Return
    ----------
        df : Dataframe with the changes made
    '''
    X_poly_features = X.drop(columns = variables_to_ignore)

    X_ignore = X[variables_to_ignore].reset_index(drop = True)

    poly = PolynomialFeatures(degree = n)

    poly_array = poly.fit_transform(X_poly_features)

    poly_features_names = poly.get_feature_names_out(X_poly_features.columns)

    X_poly_features = pd.DataFrame(poly_array, columns = poly_features_names)

    return pd.concat([X_ignore, X_poly_features], axis = 1)