
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from ds11mltoolkit.data_processing import last_columndf


import pandas as pd
import numpy as np
import pytest



def test_last_columndf():

    df = pd.DataFrame({'a':['1','2','3','4'], 
                   'b': [16,7,6,16],
                   'c':[61,57,16,36],
                   'd':['12','22','13','44'],
                   'e':['Green','Red','Blue','Yellow'],
                   'f':[1,11,23,66]})
    list_features=df.columns.to_list()
    feature=list_features[0]

    df=last_columndf(df,feature)
    list_features=df.columns.to_list()
    feature=list_features[-1]

    assert feature == 'a'




