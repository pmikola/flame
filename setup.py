from setuptools import setup, find_packages

VERSION = '0.0.4.3'
DESCRIPTION = 'Flame simulation package'
LONG_DESCRIPTION = 'Simulation engine implementation of stable fluids with extra steps for dataset generation in neural network env and test'

setup(
    name="flameEngine",
    version=VERSION,
    author="Piotr Mikolajczyk",
    author_email="<pio.mikolajczyk@email.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    readme = 'README.md',
    Homepage="https://github.com/pmikola/flame",
    install_requires=[],


    keywords=['python', 'flameEngine package','flameEngine','flameEngine'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

