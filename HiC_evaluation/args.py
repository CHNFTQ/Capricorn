# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# provide common arguments for HiC matrix evaluation
# --------------------------------------------------------

import argparse

def evaluate_parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-dir', type=str)
    parser.add_argument('--predict-resolution', type=str, default='10kb')
    parser.add_argument('--predict-prefix', type=str, default='')
    parser.add_argument('--predict-caption', type=str, default='hic')

    parser.add_argument('--target-dir', type=str, default='/data/hic_data/mat/GSE174533_1-C11-CB')
    parser.add_argument('--target-resolution', type=str, default='10kb')
    parser.add_argument('--target-prefix', type=str, default='')
    parser.add_argument('--target-caption', type=str, default='hic')

    parser.add_argument('--dataset', type=str, default='GM12878')
    parser.add_argument('--bounding', type=int, default=None, help='Only evaluate the area within the bounding distance to diagonal')
    parser.add_argument('--save-name', type=str, default=None)
    return parser