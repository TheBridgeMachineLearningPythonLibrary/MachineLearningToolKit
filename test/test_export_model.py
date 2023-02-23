import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


from ds11mltoolkit.machine_learning import export_model

from sklearn.neighbors import KNeighborsClassifier
import os

import pytest

def test_export_model():
    model = KNeighborsClassifier()
    dir_model = os.path.dirname(os.path.abspath(__file__))
    dir_model = os.path.join(dir_model, 'model')
    # dir_model = os.getcwd() + '\\model'
    # # Guardamos en una variable el directorio del script
    # dir_script = os.path.dirname(os.path.abspath(__file__))
    # # Guardamos la ubicacion del modelo
    # dir_model = os.path.join(dir_script, 'model')

    assert export_model(model, dir_model=dir_model, name_model='model_KNN_basico') == None
