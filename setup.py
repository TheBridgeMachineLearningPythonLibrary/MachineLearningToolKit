from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name = 'ds11mltoolkit',
  packages = ['ds11mltoolkit'],
  version = '1.8',
  license = 'MIT',
  description = 'Helper functions for all stages of the machine learning model building process',
  long_description = long_description,
  long_description_content_type='text/markdown',
  author = 'TheBridgeMachineLearningPythonLibrary',
  author_email = 'seenstevol@protonmail.com',
  url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit',
  download_url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit/archive/refs/tags/v_1_8.tar.gz',
  keywords = ['machine learning', 'data visualization', 'data processing', 'sklearn', 'pandas'],
  install_requires=['pandas',
                    'scipy',
                    'nltk',
                    'opencv-python-headless',
                    'scikit-image',
                    'tensorflow',
                    'keras',
                    'imblearn',
                    'scikit-learn',
                    'selenium',
                    'requests',
                    'beautifulsoup4',
                    'Pillow',
                    'matplotlib',
                    'seaborn',
                    'plotly',
                    'wordcloud',
                    'folium'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
