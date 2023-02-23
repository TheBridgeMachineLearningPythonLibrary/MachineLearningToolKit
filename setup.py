from distutils.core import setup

setup(
  name = 'ds11mltoolkit',
  packages = ['ds11mltoolkit'],
  version = '1.1',
  license = 'MIT',
  description = 'Helper functions for all stages of the machine learning model building process',
  author = 'TheBridgeMachineLearningPythonLibrary',
  author_email = 'seenstevol@protonmail.com',
  url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit',
  download_url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit/archive/refs/tags/V_1_1.tar.gz',
  keywords = ['machine learning', 'data visualization', 'data processing', 'sklearn', 'pandas'],
  install_requires=['PIL',
                    'bs4',
                    'cv2',
                    'imblearn',
                    'keras',
                    'matplotlib',
                    'nltk',
                    'numpy',
                    'pandas',
                    'plotly',
                    'scipy',
                    'seaborn',
                    'seleniumrequests',
                    'skimage',
                    'sklearn',
                    'wordcloud'],
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
