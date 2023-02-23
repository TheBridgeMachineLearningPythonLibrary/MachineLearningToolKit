import sys
sys.path.append("..")


from ds11mltoolkit.data_processing import _exponential_smooth
import pandas as pd
def test_exponential_smooth():
    # Define a sample dataset to test with
    data = pd.Series([1, 3, 4, 6, 7, 8, 9, 10])

    # Apply the exponential smoothing function with alpha=0.3
    smoothed_data = _exponential_smooth(data, alpha=0.65)

    # Check that the length of the smoothed data is the same as the original data
    assert len(smoothed_data) == len(data)

    # Check that the first and last values of the smoothed data are the same as the original data
    assert smoothed_data.iloc[0] == data.iloc[0]
    assert smoothed_data.iloc[-1] == data.iloc[-1]

    # Check that the smoothed data is less 'rigid' than the original data
    assert smoothed_data.diff().var() < data.diff().var()
