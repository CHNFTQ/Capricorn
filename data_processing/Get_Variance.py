# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to compute the channel variances and initial difficulties.
# --------------------------------------------------------

import sys
from typing import Any
import numpy as np
from Arg_Parser import *
from utils import *
from tqdm import tqdm
from data_processing.Transform import transforms, hic_normalize
from data_processing.oe_normalize import oe_normalize
from data_processing.TAD import TAD
from data_processing.insulation_score import insulation_score
from data_processing.loop_detect import HiCCUPS
from data_processing.TADaggregate import TADaggregate
import shutil 
import gc

except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}

def downsample_new(in_file, low_res, ratio):
    data = np.load(in_file)
    hic = data['hic']
    compact_idx = data['compact']
    norm = data['norm']
    down_hic = downsampling(hic, ratio)
    chr_name = os.path.basename(in_file).split('_')[0]
    out_file = os.path.join(os.path.dirname(in_file), f'{chr_name}_{low_res}.npz')
    np.savez_compressed(out_file, hic=down_hic, compact=compact_idx, norm=norm, ratio=ratio)
    print('Saving file:', out_file)
    
def transform(file, transforms):
    hic_data = np.load(file)
    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[0]

    print(f'[Chr{n}]File loaded.')

    # transform to multi-channel data
    hic = transforms(hic_data)
    
    print(f'[Chr{n}]Transformation completed.')

    return hic, compact_idx, full_size

def get_variance(save_dir, chr_list, cell_line, low_res, bound=200, seed_num=10, channel_num = 5):
    chan_var_list = [[], [], [], [], []]
    for chro in chr_list:
        abandon_chromosome = abandon_chromosome_dict[cell_line]
        if chro in abandon_chromosome:
            continue
        boundary = 0
        for chan in range(channel_num):
            a = []
            for seed in range(seed_num):
                root_dir = f'{save_dir}/{seed}'
                out_dir = os.path.join(root_dir, 'multichannel_mat', '_'.join(trs), cell_line)
                a.append(np.load(f"{out_dir}/chr{chro}_{low_res}.npz")['hic'][chan,:,:])

            if chan == 0:
                boundary = np.zeros_like(a[0], dtype=float)
                size = a[0].shape[0]
                for i in range(size):
                    boundary[i, max(0, i - bound): min(size, i + bound + 1)] = 1
            valid = np.sum(boundary)
            mean = np.mean(a, axis=0)
            var = np.zeros_like(a[0], dtype=float)
            for matrix in a:
                var += np.square((matrix - mean) * boundary) 
            var /= len(a)
            t = var.mean() * size * size / valid
            print(f"Variance for chromosome {chro} and chan {chan} is {t}")
            chan_var_list[chan].append(t)
            del a
            del mean
            del var
            if chan == channel_num - 1:
                del boundary
            gc.collect()
    print("all channel variance list", chan_var_list)
    mean_variance_list = []
    for i in range(channel_num):
        print("channel", i, np.mean(chan_var_list[i]))
        mean_variance_list.append(np.mean(chan_var_list[i]))
    print("mean variance list", mean_variance_list)
    initial_weights = [np.sqrt(mean_variance_list[0]/mean_variance_list[i]) for i in range(channel_num)]
    print('channel initial weights: ', np.array2string(initial_weights, precision=8, separator=', ',  floatmode='fixed'))

if __name__ == '__main__':
    parser = data_down_parser()
    parser.add_argument('-lrc', dest='lr_cutoff', help='REQUIRED: cutoff for low resolution maps[example:100]',
                          default=100, type=int, required=True)
    parser.add_argument('-bound', dest='bound', help='REQUIRED: distance boundary interested[example:200]',
                              default=200, type=int, required=True)
    parser.add_argument('-s', dest='dataset', help='REQUIRED: Dataset for train/valid/predict(all)',
                          default='test', choices=['train', 'valid', 'test', 'train1', 'valid1', 'test1'], )
    parser.add_argument('--seed-num', type=int, default = 10,
                        help='The number of seed used to calculate variance, default: 10')
    parser.add_argument('--old-root-dir', type=str, default = '/data/hic_data', 
                        help='The Directory stored the original matrices')
    parser.add_argument('--save-dir', type=str,
                        help='REQUIRED: The Directory to store the generated matrices')
    parser.add_argument('--transform-names', type=str, nargs='+', default = ['HiC', 'OE', '01TAD', 'Lp', 'Lr'], help='List of transforms used. Group transforms should be in the correct order(i.e. the next of Lp should be Lr)')

    args = parser.parse_args(sys.argv[1:])
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
      
    old_dir = args.old_root_dir
    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    ratio = args.ratio
    lr_cutoff = args.lr_cutoff
    bound = args.bound
    dataset = args.dataset

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

    for seed in range(args.seed_num):
        np.random.seed(seed)
        root_dir = f'{args.save_dir}/{seed}' 
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        if not os.path.isdir(os.path.join(root_dir, 'mat')):
            os.mkdir(os.path.join(root_dir, 'mat'))

        if not os.path.isdir(os.path.join(root_dir, 'mat', cell_line)):
            os.mkdir(os.path.join(root_dir, 'mat', cell_line))
        for item in os.listdir(os.path.join(old_dir, 'mat', cell_line)):
            if item.endswith(f'{high_res}.npz'):
                shutil.copyfile(os.path.join(old_dir, 'mat', cell_line, item), os.path.join(root_dir, 'mat', cell_line, item))
        
        data_dir = os.path.join(root_dir, 'mat', cell_line)
        
        in_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.find(high_res) >= 0 and f.find('_d') == -1]
        in_files = sorted(in_files)

        print(f'Generating {low_res} files from {high_res} files by {ratio}x downsampling with seed {seed}.')
        for file in in_files:
            downsample_new(file, low_res, ratio)
    
    chr_list = set_dict[dataset]
    CHANNEL_NUM = len(trs) 
    for seed in range(args.seed_num):
        np.random.seed(seed)
        root_dir = f'{args.save_dir}/{seed}'
        
        abandon_chromosome = abandon_chromosome_dict[cell_line]

        data_dir = os.path.join(root_dir, 'mat', cell_line)
        out_dir = os.path.join(root_dir, 'multichannel_mat', '_'.join(trs), cell_line)
        mkdir(out_dir)

        results = []
        for n in tqdm(chr_list):
            if n in abandon_chromosome:
                continue

            down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
            hic, compact_idx, full_size = transform(down_file, lr_transforms)
            down_save_file = os.path.join(out_dir, f'chr{n}_{low_res}.npz')
            np.savez_compressed(down_save_file, hic=hic, compacts=compact_idx, sizes=full_size)

    get_variance(args.save_dir, chr_list, cell_line, low_res, bound=args.bound, seed_num=args.seed_num, channel_num = CHANNEL_NUM)
