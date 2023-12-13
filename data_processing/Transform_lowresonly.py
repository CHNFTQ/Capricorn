# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to transform 2d HiC matrices to 3d.
# --------------------------------------------------------

import sys
from typing import Any
import numpy as np
from Arg_Parser import *
from utils import *
from tqdm import tqdm
from data_processing.oe_normalize import oe_normalize
from data_processing.TAD import TAD
from data_processing.TADaggregate import TADaggregate
from data_processing.insulation_score import insulation_score
from data_processing.loop_detect import HiCCUPS
from data_processing.Transform import *


if __name__ == '__main__':
    parser = data_divider_parser()
    parser.add_argument('--transform-names', type=str, nargs='+', default = ['HiC', 'OE', '01TAD', 'Lp', 'Lr'], help='List of transforms used. Group transforms should be in the correct order(i.e. the next of Lp should be Lr)')
    args = parser.parse_args(sys.argv[1:])

    cell_line = args.cell_line
    low_res = args.low_res
    lr_cutoff = args.lr_cutoff
    dataset = args.dataset

    bound = args.bound

    trs = args.transform_names

    lr_transforms = []

    for id, tr in enumerate(trs):
        if tr == 'HiC':
            lr_transforms.append(hic_normalize(cutoff=lr_cutoff))
        if tr == 'OE':
            lr_transforms.append(oe_normalize())
        if tr == '01TAD':
            lr_transforms.append(TAD())
        if tr == 'TADagg':
            lr_transforms.append(TADaggregate())
        if tr == 'IS':
            lr_transforms.append(insulation_score(distance_upper_bound = 2*bound, normalize=('ISN' in trs)))
        if tr == 'ISN':
            if 'IS' in tr:
                assert trs[id-1] == 'IS', print('ISN should be the next of IS')
            else:
                lr_transforms.append(insulation_score(distance_upper_bound = 2*bound, original=False))
        if tr == 'Lr':
            assert trs[id-1] == 'Lp', print('Lr should be the next of Lp')
            lr_transforms.append(HiCCUPS(distance_upper_bound = 2*bound))

    lr_transforms = transforms(lr_transforms)

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])
    print(f'Going to read {low_res} data from {cell_line}, then transform matrices with {trs}')

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    out_dir = os.path.join(root_dir, 'multichannel_mat', '_'.join(trs), cell_line)
    mkdir(out_dir)

    means_hr = []
    variances_hr = []
    means_lr = []
    variances_lr = []
    for n in tqdm(chr_list):
        if n in abandon_chromosome:
            continue

        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
        hic, compact_idx, full_size = transform(down_file, lr_transforms, n)
        down_save_file = os.path.join(out_dir, f'chr{n}_{low_res}.npz')
        np.savez_compressed(down_save_file, hic=hic, compact=compact_idx, sizes=full_size)

        elements_in_bound = []
        for i in range(-bound, bound+1):
            elements_in_bound.append(np.diagonal(hic, i, -2 ,-1))
        
        elements_in_bound = np.concatenate(elements_in_bound, axis=-1)
        mean = np.mean(elements_in_bound, axis=-1)
        var = np.var(elements_in_bound, axis=-1)
        print(f'lr means = {mean}, vars = {var}')
        means_lr.append(mean)
        variances_lr.append(var)

    print(f'lr mean={np.mean(means_lr, axis=0)}, vars={np.mean(variances_lr, axis=0)}')