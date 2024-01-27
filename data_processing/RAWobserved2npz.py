# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# the HiCARN implementation to read HiC data.
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------

import sys
import time
import multiprocessing
import numpy as np
import os
import argparse
from utils import mkdir, readcoo2mat
from dataset_informations import *

def data_read_parser():
    parser = argparse.ArgumentParser(description='Read raw data from Rao\'s Hi-C.')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:10kb]',
                           default='10kb', choices=res_map.keys())
    misc_args.add_argument('-q', dest='map_quality', help='Mapping quality of raw data[default:MAPQGE30]',
                           default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0'])
    misc_args.add_argument('-n', dest='norm_file', help='The normalization file for raw data[default:KRnorm]',
                           default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm'])

    return parser


def read_data(data_file, norm_file, out_dir, resolution):
    filename = os.path.basename(data_file).split('.')[0] + '.npz'
    out_file = os.path.join(out_dir, filename)
    try:
        HiC, norm, idx = readcoo2mat(data_file, norm_file, resolution)
    except:
        print(f'Abnormal file: {norm_file}')
    np.savez_compressed(out_file, hic=HiC, norm=norm, compact=idx)
    print('Saving file:', out_file)


if __name__ == '__main__':
    args = data_read_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    resolution = args.high_res
    map_quality = args.map_quality
    postfix = [args.norm_file, 'RAWobserved']

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    raw_dir = os.path.join(root_dir, RAW_dir, cell_line)

    norm_files = []
    data_files = []
    for root, dirs, files in os.walk(raw_dir):
        if len(files) > 0:
            if (resolution in root) and (map_quality in root):
                for f in files:
                    if (f.endswith(postfix[0])):
                        norm_files.append(os.path.join(root, f))
                    elif (f.endswith(postfix[1])):
                        data_files.append(os.path.join(root, f))

    out_dir = os.path.join(root_dir, hic_matrix_dir, cell_line)
    mkdir(out_dir)
    print(f'Start reading data, there are {len(norm_files)} files ({resolution}).')
    print(f'Output directory: {out_dir}')

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num={pool_num} for reading raw data')
    for data_fn, norm_fn in zip(data_files, norm_files):
        pool.apply_async(read_data, (data_fn, norm_fn, out_dir, res_map[resolution]))
    pool.close()
    pool.join()
    print(f'All reading processes done. Running cost is {(time.time()-start)/60:.1f} min.')