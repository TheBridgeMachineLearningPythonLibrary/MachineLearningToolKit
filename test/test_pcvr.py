import sys
sys.path.append("..")

from ds11mltoolkit.plot import plot_cumulative_variance_ratio

import pytest
import numpy as np
import matplotlib.pyplot as plt


pca_test = type('pca_test', (), {'explained_varianceratio': np.array([0.5, 0.3, 0.2])})
n = 3

def test_plot_cumulative_variance_ratio():

    plot_cumulative_variance_ratio(pca_test, n)
