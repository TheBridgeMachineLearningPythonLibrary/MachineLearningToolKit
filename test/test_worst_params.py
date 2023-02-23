import sys
sys.path.append("..")

from ds11mltoolkit.machine_learning import worst_params
import pytest
import numpy as np
from .imports_worst_params import clf_cv_results, df
from sklearn.svm import SVC
import json


@pytest.mark.parametrize(
        "input_a, expected",
        [
            (df, ("{'C': 5, 'degree': 3, 'gamma': 'auto', 'kernel': 's'}", 0.5)),
            (clf_cv_results, ("{'classifier': SVC(), 'classifier__C': 0.001}", 0.48333333))
        ]
)

def test_worst_params_multi(input_a, expected):
    assert worst_params(input_a) == expected