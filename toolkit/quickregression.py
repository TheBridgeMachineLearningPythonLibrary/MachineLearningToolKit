from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=17)

def quickregression(name):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    """
    Function to save time when doing Machine Learning models. 
    It only asks the name of the model to train and returns the scoring.

    Parameters
    ----------
    name = Name of the ML model.
           Input Example = LinearRegression
    
    Returns
    ----------
    MAE, MAPE, MSE, RMSE and R2 Scores.
    """

    # Fit of the model in the previously split X_train, y_train
    model = name()
    model.fit(X_train, y_train)
    # Predict of the model with X_test
    modpred = model.predict(X_test)
    # Scores of the model with y_test and the predict values.
    print("MAE test:", mean_absolute_error(y_test, modpred))
    print("MAPE test:", mean_absolute_percentage_error(y_test, modpred))
    print("MSE test:", mean_squared_error(y_test, modpred))
    print("RMSE test:", np.sqrt(mean_squared_error(y_test, modpred)))
    return(model.score(X_train, y_train))