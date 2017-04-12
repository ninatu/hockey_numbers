#!/bin/env python3
# -*- coding: utf-8 -*-

"""Hockey numbers recognition"""

from setuptools import setup, find_packages
from os.path import join, dirname
import hockey_numbers

setup(
    name='hockey_numbers',
    version=hockey_numbers.__version__,
    description=__doc__,
    maintainer='ninatu',
    maintainer_email='nina.tuluptseva@graphics.cs.msu.ru', 
    packages=find_packages(),
    entry_points={'console_scripts': ['create_row_frame_markup = hockey_numbers.markup.create_row_frame_markup.py']})
