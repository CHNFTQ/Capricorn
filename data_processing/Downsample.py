# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the HiCARN implementation, a script to downsample HR data to obtain LR matrices with seed control. 
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import sys
import time
import numpy as np
import os
import argparse
from utils import *
from dataset_informations import *


def data_down_parser():
    parser = argparse.ArgumentParser(description='Downsample data from high resolution data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
                          default='10kb', choices=res_map.keys(), required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:10kb_d16]',
                          default='10kb_d16', required=True)
    req_args.add_argument('-r', dest='ratio', help='REQUIRED: The ratio of downsampling[example:16]',
                          default=16, type=int, required=True)

    return parser


def downsampling(matrix, down_ratio, verbose=False):
    """
    Downsampling method.
    """
    if verbose: print(f"[Downsampling] Matrix shape is {matrix.shape}")
    tag_mat, tag_len = dense2tag(matrix)
    sample_idx = np.random.choice(tag_len, tag_len // down_ratio)
    sample_tag = tag_mat[sample_idx]
    if verbose: print(f'[Downsampling] Sampling 1/{down_ratio} of {tag_len} reads')
    down_mat = tag2dense(sample_tag, matrix.shape[0])
    return down_mat

def downsample(in_file, low_res, ratio, seed):
    np.random.seed(args.seed)
    data = np.load(in_file)
    hic = data['hic']
    compact_idx = data['compact']
    norm = data['norm']
    down_hic = downsampling(hic, ratio, seed)
    chr_name = os.path.basename(in_file).split('_')[0]
    out_file = os.path.join(os.path.dirname(in_file), f'{chr_name}_{low_res}_seed{seed}.npz')
    np.savez_compressed(out_file, hic=down_hic, compact=compact_idx, norm=norm, ratio=ratio)
    print('Saving file:', out_file)


if __name__ == '__main__':
    parser = data_down_parser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    ratio = args.ratio

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    in_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.find(high_res) >= 0 and f.find('_d') == -1]
    in_files = sorted(in_files)

    print(f'Generating {low_res} files from {high_res} files by {ratio}x downsampling.')
    start = time.time()
    for file in in_files:
        downsample(file, low_res, ratio, args.seed)
    print(f'All downsampling processes done. Running cost is {(time.time()-start)/60:.1f} min.')
