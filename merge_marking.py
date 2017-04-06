#!/usr/bin/python3
import argparse
import sys
from libs.marking import Marking

parser = argparse.ArgumentParser(description='Merge marking files')
parser.add_argument('files', type=str, nargs='*')
parser.add_argument('-o', '--output', type=str, required=True, help='file to save')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])

    marking = Marking()
    for path in args.files:
        with open(path) as fin:
            marking.addFromJson(fin)
    with open(args.output, 'w') as fout:
        marking.saveJson(fout)


