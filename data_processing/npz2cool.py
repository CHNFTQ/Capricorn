import argparse
import os
import numpy as np
from dataset_informations import *
from tqdm import tqdm
from data_processing.Read_npz import read_npz
from data_processing.Read_external_norm import read_singlechromosome_norm

import cooler
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required = True)
    parser.add_argument('--hic-caption', type=str, default = 'hic')
    parser.add_argument('--external-norm-file', type=str, 
                        default = '/data/hic_data/raw/#(CELLLINE)/10kb_resolution_intrachromosomal/#(CHR)/MAPQGE30/#(CHR)_10kb.KRnorm')
    parser.add_argument('--resolution', type=str, default='10kb')

    parser.add_argument('--bound', type=int, default=200)
    parser.add_argument('--multiple', type=int, default=255)

    parser.add_argument('-c', '--cell-line', default='GM12878')
    parser.add_argument('-s', '--dataset', default='test', choices=['train', 'valid', 'test', 'train1', 'valid1', 'test1'], )
    
    parser.add_argument('--genome_id', default = 'hg19')
    args = parser.parse_args()

    data_dir = args.data_dir
    hic_caption = args.hic_caption
    external_norm_file = args.external_norm_file
    res = args.resolution
    cell_line = args.cell_line
    dataset = args.dataset
    bound = args.bound
    multiple = args.multiple
    resolution = res_map[res.split('_')[0]]

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])

    bins = {"chrom": [], "start": [], "end": []}
    reads = {"bin1_id": [], "bin2_id": [], "count": []}
    for n in tqdm(chr_list):
        if n in abandon_chromosome:
            continue

        in_file = os.path.join(data_dir, f'chr{n}_{res}.npz')

        matrix, compact_idx, norm = read_npz(in_file, hic_caption = hic_caption, bound = bound, multiple=multiple, include_additional_channels=False)

        if external_norm_file != 'NONE':
            if '#(CHR)' in external_norm_file:
                norm = read_singlechromosome_norm(external_norm_file, n, cell_line)
            else:
                raise NotImplementedError
        
        s = len(bins['chrom'])

        for i in range(matrix.shape[0]):
            bins['chrom'].append(f'chr{n}')
            bins['start'].append(i*resolution)
            bins['end'].append((i+1)*resolution)

        for i in tqdm(range(matrix.shape[0]), leave=False):
            for j in range(i, min(i+bound, matrix.shape[1])):
                RAW = float(matrix[i][j])*norm[i]*norm[j]
                if RAW > 0:
                    reads['bin1_id'].append(s+i)
                    reads["bin2_id"].append(s+j)
                    reads["count"].append(matrix[i][j])

    cool_file = os.path.join(data_dir, f"bound{bound}_{res}.cool")
    cooler.create_cooler(cool_file, bins=pd.DataFrame.from_dict(bins), pixels=pd.DataFrame.from_dict(reads), dtypes={'count': float})
