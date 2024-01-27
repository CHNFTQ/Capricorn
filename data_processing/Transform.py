# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to transform data to include additional biological views.
# --------------------------------------------------------

import sys
from typing import Any
import numpy as np
from dataset_informations import *
from utils import *
from tqdm import tqdm
import argparse
from data_processing.biological_views.oe_normalize import oe_normalize
from data_processing.biological_views.TAD import TAD
from data_processing.biological_views.TADaggregate import TADaggregate
from data_processing.biological_views.insulation_score import insulation_score
from data_processing.biological_views.loop_detect import HiCCUPS

def data_divider_parser():
    parser = argparse.ArgumentParser(description='Transform data to include additional biological views.')
    parser.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)
    parser.add_argument('-r', dest='res', help='REQUIRED: resolution specified[example:10kb]',
                          default='10kb', required=True)
    parser.add_argument('-b', dest='bound', help='distance boundary interested[example:200]',
                              default=200, type=int)
    parser.add_argument('--cutoff', type=int, help='cutoff for high resolution maps[example: 255 for High Resolution; 100 for Low Resolution]',
                              default=255)
    parser.add_argument('-tn', '--transform-names', type=str, help='List of transforms used. Group transforms should be in the correct order(i.e. the next of Lp should be Lr)',
                        nargs='+', default = ['HiC', 'OE', '01TAD', 'Lp', 'Lr'])

    return parser

def transform(file, transforms, n):
    hic_data = np.load(file)
    compact_idx = hic_data['compact']
    norm = hic_data['norm']
    full_size = hic_data['hic'].shape[0]

    print(f'[Chr{n}]File loaded.')

    # transform to multi-channel data
    hic = transforms(hic_data)
    
    print(f'[Chr{n}]Transformation completed.')

    return hic, compact_idx, norm, full_size

#example of transform
class hic_normalize:
    def __init__(self, cutoff = 100) -> None:
        self.cutoff = cutoff
    def __call__(self, data):
        matrix = data['hic']
        out = np.minimum(matrix, self.cutoff)
        out = out / self.cutoff
        out = np.expand_dims(out, 0)
        return out

class transforms:
    def __init__(self, transform_list = []) -> None:
        self.transform_list = transform_list
    def __call__(self, data) -> Any:
        channels = []
        for transform in self.transform_list:
            channels.append(transform(data)) 
        out = np.concatenate(channels, dtype=float)
        return out

if __name__ == '__main__':
    parser = data_divider_parser()
    args = parser.parse_args()

    cell_line = args.cell_line
    res = args.res
    cutoff = args.cutoff

    bound = args.bound

    trs = args.transform_names

    t = []

    for id, tr in enumerate(trs):
        if tr == 'HiC':
            t.append(hic_normalize(cutoff=cutoff))
        if tr == 'OE':
            t.append(oe_normalize())
        if tr == '01TAD':
            t.append(TAD())
        if tr == 'TADagg':
            t.append(TADaggregate())
        if tr == 'IS':
            t.append(insulation_score(distance_upper_bound = 2*bound, normalize=('ISN' in trs)))
        if tr == 'ISN':
            if 'IS' in tr:
                assert trs[id-1] == 'IS', print('ISN should be the next of IS')
            else:
                t.append(insulation_score(distance_upper_bound = 2*bound, original=False))
        if tr == 'Lr':
            assert trs[id-1] == 'Lp', print('Lr should be the next of Lp')
            t.append(HiCCUPS(distance_upper_bound = 2*bound))

    t = transforms(t)

    chr_list = set_dict['test']
    abandon_chromosome = abandon_chromosome_dict[cell_line]
    print(f'Going to read {res} data for {cell_line}, then transform matrices with {trs}')

    data_dir = os.path.join(root_dir, hic_matrix_dir, cell_line)
    out_dir = os.path.join(root_dir, multichannel_matrix_dir, '_'.join(trs), cell_line)
    mkdir(out_dir)

    means_hr = []
    variances_hr = []

    for n in tqdm(chr_list):
        if n in abandon_chromosome:
            continue
        high_file = os.path.join(data_dir, f'chr{n}_{res}.npz')
        hic, compact_idx, norm, full_size = transform(high_file, t, n)
        high_save_file = os.path.join(out_dir, f'chr{n}_{res}.npz')
        np.savez_compressed(high_save_file, hic=hic, compact=compact_idx, norm = norm, sizes=full_size)