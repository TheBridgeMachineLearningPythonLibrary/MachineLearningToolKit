import sys
sys.path.append("..")

from ds11mltoolkit.data_processing import uniq_value

import pandas as pd
import numpy as np
import pytest







def test_uniq_value():

    df = pd.DataFrame({'a':['1','2','3','4'], 
                   'b': [16,7,6,16],
                   'c':[61,57,16,36],
                   'd':['12','22','13','44'],
                   'e':['Green','Red','Blue','Yellow'],
                   'f':[1,11,23,66]})
    
    list_features=df.columns.to_list()
    feature=list_features[0]



    df_list=df['b'].tolist()
    r=uniq_value(df_list)

    assert len(r)== 1




