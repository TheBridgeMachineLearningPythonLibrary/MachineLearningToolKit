import sys
sys.path.append('..')
from ds11mltoolkit.data_processing import ImageDataGen

def test_gen_from_directory():
    max_image_count = 100

    datagen = ImageDataGen()
    generator = datagen.gen_from_directory(
        './image_data_gen_test_data/images',
        max_image_count=max_image_count,
        batch_size=1,
        save_to_dir='./image_data_gen_test_data/image_data_gen_test_generated/'
    )

    assert len(generator) == 165
    X, y = generator.next()
    assert X.shape == (1, 256, 256, 3)
    assert y.shape == (1, 2)