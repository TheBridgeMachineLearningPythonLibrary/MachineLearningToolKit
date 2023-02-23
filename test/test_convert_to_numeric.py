import pandas as pd
from mltoolkit.data_processing import convert_to_numeric

def test_convert_to_numeric():
    
    df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4.5', '5.6', '6.7'], 'C': ['a', 'b', 'c']})
    
    
    convert_to_numeric(df, 'A')
    
    
    assert df['A'][0],int
