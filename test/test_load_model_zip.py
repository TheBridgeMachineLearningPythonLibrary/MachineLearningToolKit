import sys
sys.path.append("..")

import pickle
import zipfile
import tempfile
from ds11mltoolkit.machine_learning import load_model_zip
import os
import tempfile

def test_load_model_zip():
    # Create a temporary directory to hold the model zip file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a test model
        model = {'name': 'example_model', 'accuracy': 0.95}
        with open(os.path.join(tmp_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        # Create a zip file containing the test model
        with zipfile.ZipFile(os.path.join(tmp_dir, 'model.zip'), 'w') as zip:
            zip.write(os.path.join(tmp_dir, 'model.pkl'), 'model.pkl')

        # Load the model from the zip file
        zip_file = os.path.join(tmp_dir, 'model.zip')
        model_file = 'model.pkl'
        loaded_model = load_model_zip(zip_file, model_file)

        # Check that the loaded model matches the test model
        assert isinstance(loaded_model, dict)
        assert loaded_model['name'] == 'example_model'
        assert loaded_model['accuracy'] == 0.95
