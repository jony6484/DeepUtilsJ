from setuptools import find_packages, setup

VERSION = '0.16.0'
DESCRIPTION = 'Deep learning utils'
setup(
    name='DeepUtilsJ',
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    author='Jonathan Fuchs',
    author_email="<jony6484@gmail.com>",
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'plotly',
        'torchinfo',
        'dill'
    ],
)
