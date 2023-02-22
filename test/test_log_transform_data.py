import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import cv2 
import os
from skimage.io import imread
import sys
import pytest

sys.path.append('/home/sean/Documentos/the_bridge_bootcamp/My_Workspaces/MachineLearningToolKit/toolkit')
from data_processing import log_transform_data


def test_log_transform_data_ignore():
    df = pd.DataFrame({'a':[0, 0, 1, 0],
                  'b': [16, 7, 6, 16],
                  'c':[61, 57, 16, 36],
                  'd':['12','22','13','44'],
                  'e':['Green','Red','Blue','Yellow'],
                  'f':[1, 11, 23, 66]})

    df_processed = log_transform_data(df, col_ignore = ['a'])

    pd.testing.assert_frame_equal(df[['a', 'd', 'e']], df_processed[['a', 'd', 'e']])


def test_log_transform_data_log():
    df = pd.DataFrame({'a':[0, 0, 1, 0],
                  'b': [16, 7, 6, 16],
                  'c':[61, 57, 16, 36],
                  'd':['12','22','13','44'],
                  'e':['Green','Red','Blue','Yellow'],
                  'f':[1, 11, 23, 66]})

    df_processed = log_transform_data(df, col_ignore = ['a'])

    pd.testing.assert_frame_equal(df_processed[['b']], pd.DataFrame(np.log1p(df['b'])))