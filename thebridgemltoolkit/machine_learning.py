from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from typing import List, Union

def balance_binary_target(df, strategy='smote', minority_ratio=None, visualize=False):
    """
    This function balances a target binary variable of a dataframe using different oversampling strategies.
    Args:
    - df: dataframe with the target variable to balance.
    - strategy: oversampling strategy to use (default='smote'). The options are: 'smote', 'adasyn' or 'random'.
    - minority_ratio: proportion of the minority class after oversampling (default=None).
    - visualize: if True, visualize the balanced data (default=False).
    Returns:
    - DataFrame: dataframe with the balanced target variable.
    """

    # Automatically detect the target variable column.
    target_col = df.select_dtypes(include=['bool', 'int', 'float']).columns[0]

    # Separate target variable and predictor variables
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Oversampling the minority class using the selected strategy
    if strategy == 'smote':
        sampler = SMOTE(random_state=42)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif strategy == 'random':
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError("Estrategia de sobremuestreo inválida. Las opciones son: 'smote', 'adasyn' o 'random'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Adjust the minority class ratio if specified
    if minority_ratio is not None:
        target_counts = y_resampled.value_counts()
        minority_class = target_counts.idxmin()
        majority_class = target_counts.idxmax()

        minority_count = target_counts[minority_class]
        majority_count = target_counts[majority_class]

        desired_minority_count = int(minority_ratio * (minority_count + majority_count))

        if desired_minority_count < minority_count:
            drop_indices = y_resampled[y_resampled == minority_class].index[:minority_count - desired_minority_count]
            X_resampled = X_resampled.drop(drop_indices)
            y_resampled = y_resampled.drop(drop_indices)
        elif desired_minority_count > minority_count:
            extra_count = desired_minority_count - minority_count
            extra_X, extra_y = sampler.fit_resample(X_resampled[y_resampled == minority_class], y_resampled[y_resampled == minority_class])
            X_resampled = pd.concat([X_resampled, extra_X], axis=0)
            y_resampled = pd.concat([y_resampled, extra_y], axis=0)

    # Display the balanced data if specified
    if visualize:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Distribución de la variable objetivo balanceada')
        y_resampled.value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(target_col)
        ax.set_ylabel('Frecuencia')

    # Combine the predictor variables and the balanced target variable in a new dataframe.
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return df_resampled


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