import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


import pandas as pd
from ds11mltoolkit.data_processing import convert_to_numeric

def test_convert_to_numeric():
    
    df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4.5', '5.6', '6.7'], 'C': ['a', 'b', 'c']})
    
    
    convert_to_numeric(df, 'A')
    
    
    assert df['A'][0],int
