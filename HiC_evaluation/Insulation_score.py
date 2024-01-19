# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# our implementation of the insulation score algorithm.
# --------------------------------------------------------

import argparse
import numpy as np
import os
from HiC_evaluation.utils import *
from dataset_informations import *
from data_processing.Read_npz import read_npz
import numpy as np
import logging
from utils import compactM
import sys
import json

eps=1e-20

def compute_insulation_score(matrix, window_size, extra_channel = None):
    if len(matrix.shape) == 2:
        L, _ = matrix.shape
        scores = []
        for i in range(L):
            if i<window_size or i+window_size >= L: scores.append(None)
            else:
                scores.append(np.mean(matrix[i-window_size:i, i+1:i+window_size+1]))

    elif len(matrix.shape) == 3 and extra_channel is not None:
        # use the extra channel as score.
        C, L, _ = matrix.shape
        scores = []
        for i in range(L):
            if i<window_size or i+window_size >= L: scores.append(None)
            else:
                if matrix[extra_channel, i-window_size, i+window_size] > 0:
                    scores.append(matrix[1, i-window_size, i+window_size])
                else:
                    # extra-channel not available, use original instead
                    scores.append(np.mean(matrix[0, i-window_size:i, i+1:i+window_size+1]))
            
    return scores

def compute_boundaries(insulation_scores, delta_smooth_size, bound_strength, return_bound_strength = False):
    L = len(insulation_scores)
    
    mean_score = np.mean([s for s in insulation_scores if s is not None])
        
    normalized_scores = []
    for score in insulation_scores:
        if score is not None:
            normalized_scores.append((np.log(score+eps) - np.log(mean_score+eps))/np.log(2))
        else:
            normalized_scores.append(None)
    
    delta = []
    for i in range(L):
        left = []
        right = []
        for j in range(delta_smooth_size):
            if i+1+j<L and normalized_scores[i+1+j] is not None:
                right.append(normalized_scores[i+1+j])
            if i-1-j>=0 and normalized_scores[i-1-j] is not None:
                left.append(normalized_scores[i-1-j])
        if len(left) == 0 or len(right) == 0:
            delta.append(None)
        else:
            delta.append(np.mean(right) - np.mean(left))
        
    minimas = []
    for i in range(L):
        if i==0 or delta[i-1] is None or delta[i] is None: continue
        if delta[i-1] < 0 and delta[i] >0:
            minimas.append(i)

    # print(np.around(insulation_scores[5420:5470], decimals=4))
    # print(np.around(normalized_scores[5420:5470], decimals=4))
    # print(np.around(delta[5420:5470], decimals=4))

    bounds = []
    for minima in minimas:
        l = minima - 1
        while delta[l-1] is not None and delta[l-1] <= delta[l]:
            l -= 1
        
        r = minima + 1
        while delta[r+1] is not None and delta[r+1] >= delta[r]:
            r += 1

        if (delta[r] is None) or (delta[l] is None): continue

        if delta[r] - delta[l] >= bound_strength:
            if return_bound_strength:
                bounds.append((minima, delta[r] - delta[l]))
            else:
                bounds.append(minima)
            
        
    return bounds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, required = True)
    parser.add_argument('--hic-caption', type=str, default = 'hic')    
    parser.add_argument('--external-norm-file', type=str, 
                        default = None)
    parser.add_argument('--resolution', type=str, default='10kb')

    parser.add_argument('--bound', type=int, default=200)
    parser.add_argument('--multiple', type=int, default=255)

    parser.add_argument('-c', '--cell-line', default='GM12878')
    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--delta-smooth-size', type=int, default=10)
    parser.add_argument('--bound-strength', type=float, default=0.1)
    
    args = parser.parse_args()

    data_dir = args.data_dir
    hic_caption = args.hic_caption
    external_norm_file = args.external_norm_file
    res = args.resolution
    bound = args.bound
    multiple = args.multiple

    cell_line = args.cell_line
    dataset = args.dataset
    
    resolution = res_map[res.split('_')[0]]

    save_dir = os.path.join(data_dir, res, 'Insulation_score')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])

    TAD_counts = []
    
    scores = open(os.path.join(save_dir, 'insulation_score.bed'), 'w')
    scores.write('chrom\tstart\tend\tscore\n')

    boundaries = open(os.path.join(save_dir, 'boundaries.bed'), 'w')
    boundaries.write('chrom\tstart\tend\tscore\n')

    for n in chr_list:
        if n in abandon_chromosome:
            continue

        in_file = os.path.join(data_dir, f'chr{n}_{res}.npz')
        
        matrix, compact_idx, norm = read_npz(in_file, hic_caption=hic_caption, bound = bound, multiple=multiple, include_additional_channels=False)

        matrix = compactM(matrix, compact_idx)

        insulation_scores = compute_insulation_score(matrix, args.window_size)
        TAD_boundaries = compute_boundaries(insulation_scores, args.delta_smooth_size, args.bound_strength, return_bound_strength=True)

        TAD_counts.append(len(TAD_boundaries))
        print_info(f'chr{n}: {len(TAD_boundaries)} TAD boundaries detected.')

        for i, s in enumerate(insulation_scores):
            scores.write(f'chr{n}\t{i*resolution}\t{(i+1)*resolution}\t{s}\n')

        for b, s in TAD_boundaries:
            boundaries.write(f'chr{n}\t{b*resolution}\t{(b+1)*resolution}\t{s}\n')
    
    print_info(f'Chromosome TAD counts: {TAD_counts}')
    print_info(f'Total TAD counts: {np.sum(TAD_counts)}')

        
            
        

