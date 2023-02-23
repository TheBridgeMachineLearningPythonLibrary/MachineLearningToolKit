import sys
sys.path.append("..")

import numpy as np
import pytest
from ds11mltoolkit.plot import plot_roc_curve

@pytest.fixture()
def binary_classification_data():
    y_true = np.random.randint(low=0, high=2, size=100)
    y_pred = np.random.rand(100)
    return y_true, y_pred

def test_plot_roc_curve(binary_classification_data):
    y_true, y_pred = binary_classification_data
    plot_roc_curve(y_true, y_pred)