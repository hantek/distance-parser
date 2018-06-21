# -*- coding:utf-8 -*-
# Filename: ctb.py
# Author: hantek, hankcs

import os
import argparse
from os import listdir
from os.path import isfile, join, isdir
import nltk

from utility import make_sure_path_exists, eprint, combine_files


def convert(ctb_root, out_root):
    ctb_root = join(ctb_root, 'bracketed')
    fids = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and \
        f.endswith('.nw') or \
        f.endswith('.mz') or \
        f.endswith('.wb')]
    make_sure_path_exists(out_root)

    for f in fids:
        with open(join(ctb_root, f), 'r') as src, \
             open(join(out_root, f.split('.')[0] + '.fid'), 'w') as out:
            # encoding='GB2312'
            in_s_tag = False
            try:
                for line in src:
                    if line.startswith('<S ID=') or line.startswith('<seg id='):
                        in_s_tag = True
                    elif line.startswith('</S>') or line.startswith('</seg>'):
                        in_s_tag = False
                    elif line.startswith('<'):
                        continue
                    elif in_s_tag and len(line) > 1:
                        out.write(line)
            except:
                pass


def combine_fids(fids, out_path):
    print('Generating ' + out_path)
    files = []
    for fid in fids:
        f = 'chtb_%04d.fid' % fid
        if isfile(join(ctb_in_nltk, f)):
            files.append(f)
    with open(out_path, 'w') as out:
        combine_files(files, out, ctb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine Chinese Treebank 5.1 fid files into train/dev/test set')
    parser.add_argument("--ctb", required=True,
                        help='The root path to Chinese Treebank 5.1')
    parser.add_argument("--output", required=True,
                        help='The folder where to store the output train.txt/dev.txt/test.txt')

    args = parser.parse_args()

    ctb_in_nltk = None
    for root in nltk.data.path:
        if isdir(root):
            ctb_in_nltk = root

    if ctb_in_nltk is None:
        eprint('You should run nltk.download(\'ptb\') to fetch some data first!')
        exit(1)

    ctb_in_nltk = join(ctb_in_nltk, 'corpora')
    ctb_in_nltk = join(ctb_in_nltk, 'ctb')

    print('Converting CTB: removing xml tags...')
    convert(args.ctb, ctb_in_nltk)
    print('Importing to nltk...\n')
    from nltk.corpus import BracketParseCorpusReader, LazyCorpusLoader

    ctb = LazyCorpusLoader(
        'ctb', BracketParseCorpusReader, r'chtb_.*\.*',
        tagset='unknown')

    training = list(range(1, 270 + 1)) + list(range(440, 1151 + 1))
    development = list(range(301, 325 + 1))
    test = list(range(271, 300 + 1))

    root_path = args.output
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
    combine_fids(training, join(root_path, 'train.txt'))
    combine_fids(development, join(root_path, 'dev.txt'))
    combine_fids(test, join(root_path, 'test.txt'))
