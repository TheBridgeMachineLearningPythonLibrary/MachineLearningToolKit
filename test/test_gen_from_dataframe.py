import sys
sys.path.append('..')
from ds11mltoolkit.data_processing import ImageDataGen
import pandas as pd

dataframe = pd.read_csv('./image_data_gen_test_data/image_data_gen_test.csv')

def test_gen_from_dataframe():
    max_image_count = 100

    datagen = ImageDataGen()
    generator = datagen.gen_from_dataframe(
        dataframe,
        x_col='path',
        y_col='label',
        max_image_count=max_image_count,
        batch_size=1,
        save_to_dir='./image_data_gen_test_data/image_data_gen_test_generated/'
    )
    
    assert len(generator) == 165
    X, y = generator.next()
    assert X.shape == (1, 256, 256, 3)
    assert y.shape == (1, 2)