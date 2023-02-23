from toolkit.machine_learning import import_model

from sklearn.neighbors import KNeighborsClassifier
import os

import pytest

def test_import_model():
    model_import = KNeighborsClassifier()
    dir_model = os.path.join(os.getcwd(), 'model_KNN')
    # dir_model = os.getcwd() + '\\model_KNN'
    # # Guardamos en una variable el directorio del script
    # dir_script = os.path.dirname(os.path.abspath(__file__))
    # # Guardamos la ubicacion del modelo
    # dir_model = os.path.join(dir_script, 'model_KNN')

    assert import_model(dir_model=dir_model, name_model='model_KNN') == model_import
