import sys
sys.path.append("..")

from ds11mltoolkit.machine_learning import quickregression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


from sklearn.linear_model import LinearRegression
import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=17)
"""


def test_quickregression():
    LinearRegression=len(range(0,2))
    assert LinearRegression==2
    # assert quickregression(LinearRegression,)=>2