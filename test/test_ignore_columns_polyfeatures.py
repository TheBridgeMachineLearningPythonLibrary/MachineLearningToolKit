import sys
sys.path.append("..")

import pandas as pd
from typing import List
from sklearn.preprocessing import PolynomialFeatures

from ds11mltoolkit.machine_learning import ignore_columns_polyfeatures


def test_log_transform_data_ignore():
    df = pd.DataFrame({'a':[0, 0, 1, 0],
                  'b': [16, 7, 6, 16],
                  'c':[61, 57, 16, 36],
                  'd':['12','22','13','44'],
                  'e':['Green','Red','Blue','Yellow'],
                  'f':[1, 11, 23, 66]})

    df_processed = ignore_columns_polyfeatures(df, variables_to_ignore = ['a', 'd', 'e'], n = 2)

    pd.testing.assert_frame_equal(df[['a', 'd', 'e']], df_processed[['a', 'd', 'e']])



def test_log_transform_data_ignore():
    df = pd.DataFrame({'a':[0, 0, 1, 0],
                  'b': [16, 7, 6, 16],
                  'c':[61, 57, 16, 36],
                  'd':['12','22','13','44'],
                  'e':['Green','Red','Blue','Yellow'],
                  'f':[1, 11, 23, 66]})

    df_processed = ignore_columns_polyfeatures(df, variables_to_ignore = ['a', 'd', 'e'], n = 2)

    assert len(df_processed.columns) == 13