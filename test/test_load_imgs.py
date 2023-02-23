# Importa la función a testear
import sys
sys.path.append("..")

from ds11mltoolkit.data_processing import load_imgs


import pandas as pd
import numpy as np
import os
import pytest


def test_load_imgs():
    # Define algunos parámetros de prueba
    path = os.getcwd() + '\\img'
    im_size = 64
    
    # Llama a la función de prueba y obtiene los resultados
    df, X_train, y_train = load_imgs(path, im_size)
    
    # Comprueba si los resultados devueltos son los esperados
    assert isinstance(df, pd.DataFrame)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert df.shape[0] == len(X_train) == len(y_train)
    assert set(df.columns) == set(['filename', 'category'])
    assert set(df['category'].unique()) == set(range(len(os.listdir(path))))

