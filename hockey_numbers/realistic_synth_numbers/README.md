# Realistic synth numbers


## Preparing data

To use it you need to clone:
* https://bitbucket.org/fayao/dcnf-fcsp/
* https://github.com/jponttuset/mcg/tree/master/pre-trained

Script to prepare data(compute depth and segment map): **scripts/create_dset.py**
usage: `python create_dset.py [-h] -i [INPUT] -o [OUTPUT]` (Input hdf5 database must contain images in group "image".)

## Generating synth numbers

Uses SynthText algorithm  of [Ankush Gupta](https://github.com/ankush-me/SynthText).

File constants.py stores SYNTH_TEXT_DATA_DIR, read more about this [there](https://github.com/ankush-me/SynthText).

usage: `python2 main.py [-h] -i [INPUT] -o [OUTPUT] [-n COUNT] [--viz]`

## Extract results
Script to unpack data: **scripts/extract_to_dir.py**
To unpack the ouput hdf5 file into a directory, run
`extract_to_dir.py [-h] [h5file] [outdir]`
