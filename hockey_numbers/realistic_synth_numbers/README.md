# Realistic synth numbers


## Preparing data

To use it you need to clone:
* https://bitbucket.org/fayao/dcnf-fcsp/
* https://github.com/jponttuset/mcg/tree/master/pre-trained

Script for prepare data(compute depth and segment map): **scripts/create_dset.py**
usage: `python create_dset.py [-h] -i [INPUT] -o [OUTPUT]` (Input hdf5 database must contain images in group "image".)

## Generating synth numbers

Uses SynthText algorithm  of [Ankush Gupta](https://github.com/ankush-me/SynthText).

File constants.py stores SYNTH_TEXT_DATA_DIR, read more about this [there](https://github.com/ankush-me/SynthText).

usage: `python2 main.py [-h] -i [INPUT] -o [OUTPUT] [-n COUNT] [--viz]`


