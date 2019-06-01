""" setup.py - Main setup module for configuirng the development environment """
import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(HERE, 'README.md')).read()
VERSION = '0.1'

# Publicly Available Packages (PyPi)
INSTALL_REQUIRES = [
    'torch==1.0.0',
    'torchvision==0.2.1',
]

DEV_REQUIREMENTS = [
    'pylint==1.8.2',
    'pytest-cov==2.5.1',
    'pytest==3.6.1'
]


PROD_REQUIREMENTS = []

EXTRAS_REQUIRE = {
    'prod': PROD_REQUIREMENTS,
    'dev': DEV_REQUIREMENTS,
}

setup(
    name='OCD',
    version=VERSION,
    description="OCD Learning",
    long_description=README,
    classifiers=[
        'Programming Language :: Python :: 3.7'
    ],
    keywords="'NLP'",
    author="'Saeed Najafi'",
    author_email="'snajafi@ualberta.ca'",
    url='https://github.com/SaeedNajafi/pytorch-ocd',
    license='',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    entry_points={
        'console_scripts': [
            'ocd=ocd.__main__:main'
        ]
    },
    extras_require=EXTRAS_REQUIRE
)
