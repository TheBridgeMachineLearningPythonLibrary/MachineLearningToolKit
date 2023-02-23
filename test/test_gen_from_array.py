import sys
sys.path.append('..')
from ds11mltoolkit.data_processing import ImageDataGen
import numpy as np

x = np.load('./image_data_gen_test_data/X_image_data_gen_test.npy')
y = np.load('./image_data_gen_test_data/y_image_data_gen_test.npy')

def test_gen_from_array():
    max_image_count = 100,

    datagen = ImageDataGen()
    X_gen, y_gen = datagen.gen_from_array(
        x,
        y,
        max_image_count=max_image_count,
        batch_size=1
    )

    assert X_gen.shape == (35, 48, 48, 1)
    assert y_gen.shape == (35,)