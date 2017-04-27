#!/usr/bin/python3

from utils.markup import Markup
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('markup_file', nargs='?', help='input file with markup')
args = parser.parse_args()

markup_file = args.markup_file
out_file = osp.join(osp.dirname(markup_file), 'statistics.txt')

markup = Markup()
markup.merge(markup_file)
markup.print_statistics(out_file)


