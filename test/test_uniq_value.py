import sys
sys.path.append("..")

from toolkit.data_processing import uniq_value

import pandas as pd
import numpy as np
import pytest





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

def test_uniq_value(test_data):
    df,feature=test_data()
    df_list=df['b'].tolist()
    r=uniq_value(df_list)

    assert len(r)== '3'



