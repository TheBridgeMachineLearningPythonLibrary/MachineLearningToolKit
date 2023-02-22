import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union


def plot_multiclass_prediction_image(df, row_index: int, X_test: Union[pd.DataFrame, np.ndarray], prediction_col: str = 'Top Prediction', label_col: str = 'Label'):
    '''
    This will produce two plots side by side relating to a prediction of an image
    The first is the image titled with the label and the predicted label
    The second is a bar plot showing the actual probabilities of predictions for all classes

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the results of the prediction along with the label and top predicted label
    row_index : int
        the index value for the image we want to plot
    X_test : pd.DataFrame or np.ndarray
        The X_test data
    prediction_col : str
        Defaults to 'Top Prediction' but should be the name of column with the prediction for image
    label_col : str
        Defaults to 'Label' but should be the name of column with the real label
    
    Return
    ----------
    displays plots

    '''

    to_plot = df.loc[[row_index]]

    fig, ax = plt.subplots(1, 2, figsize = (10, 5), constrained_layout = True)

    ax[0].imshow(X_test[row_index])
    ax[0].set_title(f'Label: {df.loc[row_index, prediction_col]}\nPrediction: {df.loc[row_index, label_col]}')

    df.loc[row_index, df.dtypes != object].plot.bar(ax = ax[1])
    plt.xticks(rotation = 45, ha = 'right')
    plt.title('Probabilities of Each Class')
    plt.show()