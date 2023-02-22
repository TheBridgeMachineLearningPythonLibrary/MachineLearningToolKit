from distutils.core import setup

setup(
  name = 'thebridgemltoolkit',
  packages = ['thebridgemltoolkit'],
  version = '0.4',
  license = 'MIT',
  description = 'Helper functions for all stages of the machine learning model building process',
  author = 'TheBridgeMachineLearningPythonLibrary',
  author_email = 'seenstevol@protonmail.com',
  url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit',
  download_url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit/archive/refs/tags/v_04.tar.gz',
  keywords = ['machine learning', 'data visualization', 'data processing', 'sklearn', 'pandas'],
  install_requires=[
          'pandas',
          'numpy',
          'imblearn',
          'sklearn',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)