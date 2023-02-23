import sys
sys.path.append("..")

import pytest
import pandas as pd
from ds11mltoolkit.data_analysis import read_url

def test_read_url():

    # Test case 1: Valid URL and correct data format
    url1 = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'
    df1 = read_url(url1)
    assert isinstance(df1, pd.DataFrame)

    # Test case 2: Valid URL but incorrect data format
    url2 = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    with pytest.raises(Exception):
        read_url(url2)

    # Test case 3: Invalid URL
    url3 = 'https://thisurlisnotvalid.com'
    with pytest.raises(Exception):
        read_url(url3)