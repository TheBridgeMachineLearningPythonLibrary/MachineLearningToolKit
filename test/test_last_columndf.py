

from toolkit.data_processing import last_columndf, uniq_value,list_categorical_columns
from toolkit.data_analysis import *
from toolkit.machine_learning import *
from toolkit.plot import *

import pandas as pd
import numpy as pd
import pytest


@pytest.fixture


def test_data():

    df = pd.DataFrame({'a':['1','2','3','4'], 
                   'b': [16,7,6,16],
                   'c':[61,57,16,36],
                   'd':['12','22','13','44'],
                   'e':['Green','Red','Blue','Yellow'],
                   'f':[1,11,23,66]})
    list_features=df.columns.to_list()
    feature=list_features[0]
    return df,feature


def test_last_columndf(test_data):
    df,feature=test_data()
    df=last_columndf(df,feature)
    list_features=df.columns.to_list()
    feature=list_features[-1]

    assert feature == 'a'



