import sys
sys.path.append("..")

from ds11mltoolkit.machine_learning import processing_model_classification

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import pytest

def test_processing_model_classification():
    model = KNeighborsClassifier()
    df = pd.DataFrame({'sex': [0, 1, 1, 0, 1, 1, 0], 'hypertension': [0, 0, 1, 1, 0, 0, 1], 'heart_disease': [0, 0, 1, 1, 0, 1, 1], 'Residence_type': [1, 1, 0, 1, 0, 1, 0], 'stroke': [0, 1, 1, 0, 1, 0, 1]})
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    X_train = pd.DataFrame({'sex':[0, 1, 1, 1, 0], 
                   'hypertension': [0, 1, 0, 0, 1],
                   'heart_disease':[0, 1, 0, 1, 1],
                   'Residence_type':[1, 0, 0, 1, 1]}
                   )
    X_test = pd.DataFrame({'sex':[0, 1], 
                   'hypertension': [1, 0],
                   'heart_disease':[1, 0],
                   'Residence_type':[0, 1]}
                   )
    y_train = pd.DataFrame({'':[0, 1, 1, 0, 0]})
    y_test = pd.DataFrame({'':[1, 1]})
    y_pred_train = [0, 0, 0, 0, 0] # [0 0 0 0 0]
    y_pred_test = [0, 0] # [0 0]

    model_r, X_train_r, X_test_r, y_train_r, y_test_r, y_pred_train_r, y_pred_test_r = processing_model_classification(model, X, y, test_size_split=0.20, shuffle_split=True, 
                                                                                                         random_state_split=32, standardScaler=False, train_score=True, test_score=False)

    assert str(model_r) == str(model)
    # assert X_train_r.equal(X_train)
    # assert X_test_r.equal(X_test)
    # assert y_train_r.equal(y_train)
    # assert y_test_r.equal(y_test)
    # assert y_pred_train_r == y_pred_train
    # assert y_pred_test_r == y_pred_test
