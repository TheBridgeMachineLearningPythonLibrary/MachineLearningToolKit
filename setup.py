from distutils.core import setup

setup(
  name = 'toolkit',
  packages = find_packages(),
  version = '0.1',
  license = 'MIT',
  description = 'Helper functions for all stages of the machine learning model building process',
  author = 'TheBridgeMachineLearningPythonLibrary',
  author_email = 'seenstevol@protonmail.com',
  url = 'https://github.com/TheBridgeMachineLearningPythonLibrary/MachineLearningToolKit',
  download_url = '',    # I explain this later on
  keywords = ['machine learning', 'data visualization', 'data processing', 'sklean', 'pandas'],
  install_requires=[            # I get to this in a second
          ''
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Data Scientists',
    'Topic :: Data Science :: Machine Learning Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)