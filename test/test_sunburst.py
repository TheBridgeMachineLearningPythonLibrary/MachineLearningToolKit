import sys
sys.path.append("..")

from ds11mltoolkit.plot import sunburst
import plotly.graph_objs as go
import plotly.express as px

def test_sunburst():
    df = {
        'Category': ['Fruit', 'Fruit', 'Vegetable', 'Vegetable', 'Vegetable'],
        'Subcategory': ['Apple', 'Orange', 'Carrot', 'Tomato', 'Cucumber'],
        'Value': [20, 30, 40, 15, 25] }
    
    fig = sunburst(df, 'Category', 'Subcategory', 'Value', 'My Sunburst Chart')   
    
    assert fig.layout.title.text == 'My Sunburst Chart'
    assert fig.layout.width == 800
    assert fig.layout.height == 600
