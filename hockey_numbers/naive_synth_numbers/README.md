#Improved Naive Synth Numbers Algorithm

Uses SynthText algorithm  of [Ankush Gupta](https://github.com/ankush-me/SynthText).

File constants.py stores SYNTH_TEXT_DATA_DIR, read more about this [there](https://github.com/ankush-me/SynthText).

Usage: `python2 main.py [-h] -i [INPUTDB] -o [OUTPUTDB] [-n COUNT] [--viz]`

Struct input hdf5 dataset:

dataset:
- data:
    - img1: img_dset1
        ...
    - imgN: img_dsetN
