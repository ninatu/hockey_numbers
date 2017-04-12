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
    #install_requires=['opencv3.1.0, scipy, tqdm, h5py, pillow, sklearn'],
    entry_points={'console_scripts': ['save_blobs_to_hdf5 = hockey_numbers.markup.save_blobs_to_hdf5:main',
                                      'save_frames_to_dir = hockey_numbers.markup.save_frames_to_dir:main']})
