
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


from ds11mltoolkit.data_analysis import chi_squared_test
import pandas as pd
import scipy.stats as stats
import pytest
import numpy as np


def test_chi_squared_test():

    # Create a sample DataFrame
    data = {'feature': ['A', 'B', 'C', 'D', 'E'],
            'target': np.random.randint(2, size=5)}
    df = pd.DataFrame(data)

    # Perform the chi-squared test
    chi2, p = chi_squared_test(df, 'feature', 'target')

    # Verify the results
    assert isinstance(chi2, float)
    assert isinstance(p, float)
    assert chi2 == pytest.approx(5.0, rel=1e-3)
    assert p == pytest.approx(0.287, rel=0.2872974951836458)


    



