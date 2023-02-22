from toolkit.machine_learning import quickregression

from sklearn.linear_model import LinearRegression
import pytest

"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=17)
"""

def test_quickregression():
    assert quickregression(LinearRegression) >= 0
