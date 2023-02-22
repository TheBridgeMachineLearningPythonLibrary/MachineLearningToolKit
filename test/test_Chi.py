


def test_chi_squared_test():
    from Chi import chi_squared_test
    import pandas as pd
    import scipy.stats as stats
    import pytest
    import numpy as np
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


    



