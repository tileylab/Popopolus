from setuptools import setup, find_packages

with open('README.md','r') as fh:
    long_description = fh.read()

setup(
    author="George P. Tiley",
    description="A package for polyploid popoulation genomics analyses and data exploration",
    name="popopolus",
    version="0.1.0",
    license='MIT',
    url='https://github.com/gtiley/popopolus',
    py_modules = ['popopolus_cli'],
    packages=find_packages(include=["popopolus","popopolus.*"]),
    python_requires=">=3.11",
    install_requires=[
        'cyvcf2>=0.31.1',
        'pandas>=2.2.2',
        'scikit-learn>=1.5.1,<1.8',
        'click>=8.1.7',
        'matplotlib>=3.10.5',
        'scipy>=1.15.2',
        'seaborn>=0.13.2',
        'statsmodels>=0.14.5'
    ],
    entry_points = '''
        [console_scripts]
        popopolus=popopolus_cli:cli
    '''
)
