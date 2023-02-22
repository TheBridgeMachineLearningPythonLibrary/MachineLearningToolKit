from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from typing import List, Union


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


def create_multiclass_prediction_df(model, class_names: List[str], X_test: Union[pd.DataFrame, np.ndarray], y_test: Union[list, np.ndarray], only_wrong: bool = False) -> pd.DataFrame:
    '''
    This will generate a dataframe from the predictions of a model on X_test for easy analysis of model performance

    Parameters
    ----------
    model : keras model
        It is the dataset we want to selectively create polynomial features
    class_names : List[str]
        a list of the class names which must be in order that relates to the numbers in y_test
    X_test : pd.DataFrame or np.ndarray
        The data for X_test
    y_test : list or np.ndarray
        The y_test with the numeric values of the class names
    only_wrong : bool
        defaults to False but if set to True, only incorrect predictions will be kept
    
    Return
    ----------
        df : Dataframe containing the predictions from model.predict along with the top prediction and actual labels
    
    '''
    # make the predictions with model on X_test
    model_predictions_df = round(pd.DataFrame(model.predict(X_test)), 2)
    # assign class label names as columns
    model_predictions_df.columns = class_names
    # create top prediction column
    model_predictions_df['Top Prediction'] = model_predictions_df.apply(lambda x: x.idxmax(), axis = 1)
    # create class label column
    model_predictions_df['Label'] = [model_predictions_df.columns[y_class] for y_class in y_test]
    # filter for model errors if "only_wrong == True"
    if only_wrong:
        model_predictions_df = model_predictions_df[model_predictions_df['Top Prediction'] != model_predictions_df['Label']]
    # return the dataframe
    return model_predictions_df   