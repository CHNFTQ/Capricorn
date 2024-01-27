# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to help convert .hic files to .npz files.
# --------------------------------------------------------

import sys
import os
import argparse
from dataset_informations import *
from utils import *

def hic_data_read_parser():
    parser = argparse.ArgumentParser(description='Read data from .hic files.')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-f', dest='hic_file', help='REQUIRED: hic filename', required=True)
    req_args.add_argument('-name', dest='name', help='REQUIRED: cellline name', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:10kb]',
                           default='10kb', choices=res_map.keys())
    misc_args.add_argument('-n', dest='norm_type', help='The normalization type chose[default:KRnorm]',
                           default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm', 'Raw'])

    return parser

import hicstraw
import numpy as np
from tqdm import tqdm

norm_map = {'KRnorm' : 'KR', 'SQRTVCnorm' : 'VC_SQRT', 'VCnorm' : 'VC', 'Raw' : 'NONE'}

if __name__ == '__main__':
    args = hic_data_read_parser().parse_args(sys.argv[1:])

    hic = hicstraw.HiCFile(args.hic_file)
    print(hic.getChromosomes())
    print(hic.getGenomeID())
    print(hic.getResolutions())

    out_dir = os.path.join(root_dir, hic_matrix_dir, args.name)
    mkdir(out_dir)

    chr_list = hic.getChromosomes()
    for ch in chr_list[2:]:
        print(ch.index, ch.name)
        resolution = res_map[args.high_res]
        mzd = hic.getMatrixZoomData(ch.name, ch.name, "observed", norm_map[args.norm_type], "BP", resolution)
        
        norm = mzd.getNormVector(ch.index)
        compact_idx = list(np.where(np.isnan(norm) ^ True)[0])
        norm[np.isnan(norm)] = 1

        step = int(1e7)
        N = ch.length

        HiC = np.zeros((N//resolution + 1, N//resolution + 1))

        for i in tqdm(range(0, N, step)):
            for j in tqdm(range(0, N, step), leave=False):
                submat = mzd.getRecordsAsMatrix(i, min(i+step, N), j, min(j+step, N))
                # print(np.sum(submat), submat.shape, i//resolution, j//resolution)
                HiC[i//resolution:i//resolution + submat.shape[0], j//resolution:j//resolution + submat.shape[1]] = submat
                

        filename = f'chr{ch.name}_{args.high_res}.npz'
        out_file = os.path.join(out_dir, filename)

        np.savez_compressed(out_file, hic=HiC, norm=norm, compact=compact_idx)



