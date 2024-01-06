# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to compute the channel variances and initial difficulties.
# --------------------------------------------------------

import sys
from typing import Any
import numpy as np
from dataset_informations import *
from utils import *
from tqdm import tqdm
import argparse 
import gc
    
def channel_weights_parser():
    parser = argparse.ArgumentParser(description='Calculate initial weights for channels')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
                          default='10kb_d16', required=True)
    req_args.add_argument('-s', dest='dataset', help='REQUIRED: Dataset for train/valid/predict(all)',
                          default='train', choices=set_dict.keys() )
    req_args.add_argument('-trs', '--transform-names', dest='transform_names', type=str, help='List of transforms used. Group transforms should be in the correct order(i.e. the next of Lp should be Lr)',
                          nargs='+', default = ['HiC', 'OE', '01TAD', 'Lp', 'Lr'], required=True)
    req_args.add_argument('-sd', '--seeds', dest='seeds', type=str, help='List of seeds used. ',
                          nargs='+', default = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], required=True)
    req_args.add_argument('--bound', dest='bound', help='REQUIRED: distance boundary interested[example:201]',
                              default=201, type=int, required=True)

    return parser

if __name__ == '__main__':
    parser = channel_weights_parser()
    args = parser.parse_args()
    
    cell_line = args.cell_line
    low_res = args.low_res
    dataset = args.dataset
    trs = args.transform_names
    seeds = args.seed
    bound = args.bound
    channel_num = len(trs)

    chan_var_list = [[] for _ in range(channel_num)]
    chr_list = set_dict[args.dataset]
    for chro in chr_list:
        abandon_chromosome = abandon_chromosome_dict[cell_line]
        if chro in abandon_chromosome:
            continue
        boundary = 0
        for chan in range(channel_num):
            a = []
            for seed in seeds:
                out_dir = os.path.join(root_dir, 'multichannel_mat', '_'.join(trs), cell_line)
                a.append(np.load(f"{out_dir}/chr{chro}_{low_res}_{seed}.npz")['hic'][chan,:,:])

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
