from distutils.core import setup

setup(
  name = 'mltoolkit',
  packages = ['mltoolkit'],
  version = '1.0',
  license = 'MIT',
  description = 'Helper functions for all stages of the machine learning model building process',
  author = 'TheBridgeMachineLearningPythonLibrary',
  author_email = 'seenstevol@protonmail.com',
  url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit',
  download_url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit/archive/refs/tags/v_01.tar.gz',
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
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Natural Language Processing'
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
