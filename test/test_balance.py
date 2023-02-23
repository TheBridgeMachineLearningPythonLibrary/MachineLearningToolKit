
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


from ds11mltoolkit.machine_learning import balance_binary_target



import pytest
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import math

def test_balance_binary_target_min_ratio():
    # Create a sample DataFrame with a binary target variable.
    data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}
    df = pd.DataFrame(data)

    # Perform the oversampling using SMOTE and minority_ratio of 0.4
    df_resampled = balance_binary_target(df, strategy='smote', minority_ratio=0.4)

    # Verify that the resulting DataFrame has the expected ratio of minority samples
    ratio = df_resampled['target'].mean()
    assert math.isclose(ratio, 0.4, rel_tol=1)
