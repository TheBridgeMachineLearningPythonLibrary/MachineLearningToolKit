import sys
sys.path.append("..")

from ds11mltoolkit.machine_learning import predict_model_classification, import_model

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os

import pytest

def test_predict_model_classification():
    dir_model = os.path.dirname(os.path.abspath(__file__))
    dir_model = os.path.join(dir_model, 'model')
    model_import=import_model(dir_model=dir_model, name_model='model_KNN_train')

    X_test = pd.DataFrame({'sex':[0, 1], 
                   'hypertension': [1, 0],
                   'heart_disease':[1, 0],
                   'Residence_type':[0, 1]}
                   )
    y_test = pd.DataFrame({'':[1, 1]})
    y_pred_test = [0, 0]

    X_test_model, y_pred_test_model=predict_model_classification(model_import, X_test, y_test, test_score=False) 
    
    list_y=list(y_pred_test_model.tolist())

    assert X_test_model.equals(X_test)
    assert list_y == list(y_pred_test)
    # assert y_pred_test_model.tolist() == y_pred_test
