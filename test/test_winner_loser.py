import sys
sys.path.append("..")


from ds11mltoolkit.data_processing import winner_loser
import pandas as pd
import pytest

# Define test data
df = pd.DataFrame(data= {'Goals': [0, 2, 1, 4, 2, 2], 
                            'Teams': ['Team 1', 'Team 2', 'Team 1', 'Team 2', 'Team 1', 'Team 2']})

# Run tests

@pytest.mark.parametrize(
    "input_a, input_b ,input_c, expected",
        [
            (0, df, 'Goals', 'Loss'),
            (1, df, 'Goals', 'Victory'),
            (2, df, 'Goals', 'Loss'),
            (3, df, 'Goals', 'Victory'),
            (4, df, 'Goals', 'Draw'),
            (5, df, 'Goals', 'Draw')
        ]
)

def test_winner_loser_multi(input_a, input_b, input_c, expected):
    assert winner_loser(input_a, input_b, input_c) == expected