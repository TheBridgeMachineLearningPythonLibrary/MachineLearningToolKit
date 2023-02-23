from mltoolkit.data_processing import *
from mltoolkit.data_analysis import *
from mltoolkit.machine_learning import *
from mltoolkit.plot import plot_ngrams

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pytest


def test_plot_ngrams():
    df = pd.DataFrame({'target': [1, 0, 1], 'text': ['foo bar foo', 'baz qux', 'foo bar baz']})
    n_gram = 2
    fig = plot_ngrams(df, 'target', 'text', n_gram)
    assert isinstance(fig, go.Figure)
    assert len(fig['data']) == 2

