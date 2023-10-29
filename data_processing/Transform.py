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


except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}

def transform(file, transforms):
    hic_data = np.load(file)
    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[0]

    print(f'[Chr{n}]File loaded.')

    # transform to multi-channel data
    hic = transforms(hic_data)
    
    print(f'[Chr{n}]Transformation completed.')

    return hic, compact_idx, full_size

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
    parser.add_argument('-hrc', dest='hr_cutoff', help='cutoff for high resolution maps[example:255]',
                          default=255, type=int)
    parser.add_argument('--transform-names', type=str, nargs='+', default = ['HiC', 'OE', '01TAD', 'Lp', 'Lr'], help='List of transforms used. Group transforms should be in the correct order(i.e. the next of Lp should be Lr)')
    args = parser.parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    lr_cutoff = args.lr_cutoff
    hr_cutoff = args.hr_cutoff
    dataset = args.dataset

    bound = args.bound

    trs = args.transform_names

    hr_transforms = []
    lr_transforms = []

    for id, tr in enumerate(trs):
        if tr == 'HiC':
            hr_transforms.append(hic_normalize(cutoff=hr_cutoff))
            lr_transforms.append(hic_normalize(cutoff=lr_cutoff))
        if tr == 'OE':
            hr_transforms.append(oe_normalize())
            lr_transforms.append(oe_normalize())
        if tr == '01TAD':
            hr_transforms.append(TAD())
            lr_transforms.append(TAD())
        if tr == 'TADagg':
            hr_transforms.append(TADaggregate())
            lr_transforms.append(TADaggregate())
        if tr == 'IS':
            hr_transforms.append(insulation_score(distance_upper_bound = 2*bound, normalize=('ISN' in trs)))
            lr_transforms.append(insulation_score(distance_upper_bound = 2*bound, normalize=('ISN' in trs)))
        if tr == 'ISN':
            if 'IS' in tr:
                assert trs[id-1] == 'IS', print('ISN should be the next of IS')
            else:
                hr_transforms.append(insulation_score(distance_upper_bound = 2*bound, original=False))
                lr_transforms.append(insulation_score(distance_upper_bound = 2*bound, original=False))
        if tr == 'Lr':
            assert trs[id-1] == 'Lp', print('Lr should be the next of Lp')
            hr_transforms.append(HiCCUPS(distance_upper_bound = 2*bound))
            lr_transforms.append(HiCCUPS(distance_upper_bound = 2*bound))

    hr_transforms = transforms(hr_transforms)
    lr_transforms = transforms(lr_transforms)

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict[cell_line]
    print(f'Going to read {high_res} and {low_res} data, then transform matrices with {trs}')

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
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        hic, compact_idx, full_size = transform(high_file, hr_transforms)
        high_save_file = os.path.join(out_dir, f'chr{n}_{high_res}.npz')
        np.savez_compressed(high_save_file, hic=hic, compact=compact_idx, sizes=full_size)

        elements_in_bound = []
        for i in range(-bound, bound+1):
            elements_in_bound.append(np.diagonal(hic, i, -2 ,-1))
        
        elements_in_bound = np.concatenate(elements_in_bound, axis=-1)
        mean = np.mean(elements_in_bound, axis=-1)
        var = np.var(elements_in_bound, axis=-1)
        print(f'hr means = {mean}, vars = {var}')
        means_hr.append(mean)
        variances_hr.append(var)

        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
        hic, compact_idx, full_size = transform(down_file, lr_transforms)
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

    print(f'hr mean={np.mean(means_hr, axis=0)}, vars={np.mean(variances_hr, axis=0)}')
    print(f'lr mean={np.mean(means_lr, axis=0)}, vars={np.mean(variances_lr, axis=0)}')