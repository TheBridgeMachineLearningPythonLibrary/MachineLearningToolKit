import sys
sys.path.append("..")


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ds11mltoolkit.plot import heatmap

def test_heatmap():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'target': [10, 20, 30]})

    heatmap(df, 2, 'target', None)
    

    
