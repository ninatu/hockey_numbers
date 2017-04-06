#!/usr/bin/python3
import argparse
import sys
from libs.marking import Marking

parser = argparse.ArgumentParser(description='Merge marking files')
parser.add_argument('-i', '--input', type=str, required=True, help='input file with marking')
parser.add_argument('-o', '--output', type=str, required=True, help='file to save')


if __name__ == '__main__':
    args = parser.parse_args()

    marking = Marking()
    with open(args.input) as fin:
        marking.addFromJson(fin)
    marking.filterByField(0, 820, 0, 5700)
    with open(args.output, 'w') as fout:
        marking.saveJson(fout)



