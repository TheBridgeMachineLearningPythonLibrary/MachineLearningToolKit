import sys
sys.path.append("..")

from ds11mltoolkit.data_processing import list_categorical_columns


import pandas as pd
import numpy as np
import pytest




def test_list_categorical_columns():
    df = pd.DataFrame({'a':['1','2','3','4'], 
                   'b': [16,7,6,16],
                   'c':[61,57,16,36],
                   'd':['12','22','13','44'],
                   'e':['Green','Red','Blue','Yellow'],
                   'f':[1,11,23,66]})
    
    list_features=df.columns.to_list()
    

    r=list_categorical_columns(df)
       

    assert r== ['a', 'd', 'e']



