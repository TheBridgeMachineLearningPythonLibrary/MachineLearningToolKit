import sys
sys.path.append("..")

import pandas as pd
import os
from io import BytesIO
from ds11mltoolkit.data_analysis import read_csv_zip
import zipfile


def test_read_csv_zip():
    # Create a CSV file
    csv_content = f"name;age;city\nJohn;30;Madrid\nSara;28;London\n"
    csv_file = "test.csv"
    with BytesIO() as buffer:
        buffer.write(csv_content.encode())
        buffer.seek(0)
        with zipfile.ZipFile('test.zip', 'w') as myzip:
            myzip.writestr(csv_file, buffer.read())

    # Test case 1: read CSV file with default separator
    zip_file = 'test.zip'
    df = read_csv_zip(zip_file, csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert df.iloc[0, 0] == 'John'
    assert df.iloc[1, 2] == 'London'

    # Test case 2: read CSV file with custom separator
    df = read_csv_zip(zip_file, csv_file, sep=';')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert df.iloc[0, 0] == 'John'
    assert df.iloc[1, 2] == 'London'

    # Clean up
    os.remove(zip_file)
