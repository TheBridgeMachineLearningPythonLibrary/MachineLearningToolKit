
import pandas as pd
import pytest

from mltoolkit.data_processing import buffdescribe
from mltoolkit.data_analysis import *
from mltoolkit.machine_learning import *
from mltoolkit.plot import *

@pytest.fixture

def test_df():
    return pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [4.5, 6.7, 8.9]})

def test_buffdescribe_columns(test_df):
    result = buffdescribe(test_df)
    assert result.index.tolist() == ['a', 'b', 'c']

def test_buffdescribe_data_type(test_df):
    result = buffdescribe(test_df)
    assert result['DATA_TYPE'].tolist() == ['int64', 'object', 'float64']

def test_buffdescribe_null_values(test_df):
    result = buffdescribe(test_df)
    assert result['MISSINGS (%)'].tolist() == [0.0, 0.0, 0.0]

def test_buffdescribe_non_null_values(test_df):
    result = buffdescribe(test_df)
    assert result['NOT_NULL'].tolist() == [3, 3, 3]

def test_buffdescribe_unique_values(test_df):
    result = buffdescribe(test_df)
    assert result['UNIQUE_VALUES'].tolist() == [3, 3, 3]

def test_buffdescribe_cardinality(test_df):
    result = buffdescribe(test_df)
    assert result['CARDIN (%)'].tolist() == [100.0, 100.0, 100.0]

def test_buffdescribe_descriptive_statistics(test_df):
    result = buffdescribe(test_df)
    assert result.columns.tolist() == ['DATA_TYPE', 'MISSINGS (%)', 'NOT_NULL', 'UNIQUE_VALUES', 'CARDIN (%)',
                                       'mean', 'median', 'std']

