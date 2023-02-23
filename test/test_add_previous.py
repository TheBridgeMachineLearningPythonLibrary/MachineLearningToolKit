from mltoolkit.data_processing import buffdescribe
from mltoolkit.data_analysis import *
from mltoolkit.machine_learning import *
from mltoolkit.plot import *
import pandas as pd

df = pd.DataFrame(data= {'Goals': [0, 2, 1, 4, 5, 2, 1, 0, 2, 0, 1, 1], 
                            'Teams': ['Team 1', 'Team 2', 'Team 1', 'Team 2', 'Team 1', 'Team 2',
                                        'Team 1', 'Team 2', 'Team 1', 'Team 2', 'Team 1', 'Team 2']})


def test_add_previous():
    add_previous(df, 3, 'Teams', 'Goals')
    list_features = df.columns.to_list()
    print(list_features)
    length = len(list_features)

    assert length == 5