import pytest
import pandas as pd

from mltoolkit.data_processing import clean_text
from mltoolkit.data_analysis import *
from mltoolkit.machine_learning import *
from mltoolkit.plot import *

@pytest.fixture

def test_data():
    # Crea un conjunto de datos de prueba para la función clean_text
    data = {'text': ['Este es un texto de prueba', 'Otro texto de prueba'],
            'target': [0, 1]}
    df = pd.DataFrame(data)
    return df

def test_clean_text(test_data):
    # Ejecuta la función clean_text con el conjunto de datos de prueba
    result = clean_text(test_data, 'text', 'spanish', 'target', 'test.csv')
    expected_columns = ['text', 'target']
    expected_result = pd.DataFrame({'text': ['text prueb', 'text prueb'], 'target': [0, 1]})
    
    # Verifica que el resultado es igual al esperado
    assert result.columns.tolist() == expected_columns
    assert result.equals(expected_result)
