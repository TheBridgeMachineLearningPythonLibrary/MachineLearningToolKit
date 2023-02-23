from mltoolkit.machine_learning import predict_model_classification

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import pytest

def test_predict_model_classification():
    model_import = KNeighborsClassifier()
    X_test = pd.DataFrame({'sex':[0, 1], 
                   'hypertension': [1, 0],
                   'heart_disease':[1, 0],
                   'Residence_type':[0, 1]}
                   )
    y_test = pd.DataFrame({'':[1, 1]})
    y_pred_test = [0, 0] # [0 0]

    assert predict_model_classification(model_import, X_test, y_test, test_score=False) == X_test, y_pred_test

    # X_test_r, y_pred_test_r = predict_model_classification(model_import, X_test, y_test, test_score=False)
    # assert X_test_r == X_test
    # assert y_pred_test_r.tolist() == y_pred_test
