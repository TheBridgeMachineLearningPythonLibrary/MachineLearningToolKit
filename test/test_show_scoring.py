import sys
sys.path.append("..")

from ds11mltoolkit.machine_learning import show_scoring
import pandas as pd

import pytest

def test_show_scoring():
    y_test = pd.DataFrame({'':[1, 1]})
    y_pred_test = [0, 0] # [0 0]

    dict_scoring = show_scoring(y_test, y_pred_test, label='TEST', round=1, auc_sc=False, roc_auc_sc=False, confusion_matrix_sc=False)

    assert dict_scoring['accu_r'] == 0.0
    assert dict_scoring['auc_r'] == None
    assert dict_scoring['roc_auc_r'] == None
    assert dict_scoring['conf_mat_r'] == None
