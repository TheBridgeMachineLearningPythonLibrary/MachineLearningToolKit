import sys
sys.path.append("..")

from ds11mltoolkit.plot import wordcloudviz

import pytest
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

data = {'column': ['hola mundo', 'esto es una prueba', 'pytest es un engorro', 'hello world', 'tralala']}
df = pd.DataFrame(data)

def test_wordcloudviz():
    wordcloudviz(df['column'])