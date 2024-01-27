# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to help convert .npz files to .hic files.
# --------------------------------------------------------

import argparse
import os
import numpy as np
from dataset_informations import *
from tqdm import tqdm
from data_processing.Read_npz import read_npz
from data_processing.Read_external_norm import read_singlechromosome_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required = True)
    
    parser.add_argument('--norm-name', type=str, default = 'NONE')
    parser.add_argument('--external-norm-file', type=str, 
                        default = f'{root_dir}/{RAW_dir}/#(CELLLINE)/10kb_resolution_intrachromosomal/#(CHR)/MAPQGE30/#(CHR)_10kb.KRnorm')

    parser.add_argument('--resolution', type=str, default='10kb')

    parser.add_argument('--bound', type=int, default=200)
    parser.add_argument('--multiple', type=int, default=255)

    parser.add_argument('-c', '--cell-line', default='GM12878')
    parser.add_argument('-s', '--dataset', default='test', choices=['train', 'valid', 'test', 'train1', 'valid1', 'test1'], )
    
    parser.add_argument('--genome-id', default = 'hg19')
    args = parser.parse_args()

    data_dir = args.data_dir
    norm_name = args.norm_name
    external_norm_file = args.external_norm_file
    res = args.resolution
    cell_line = args.cell_line
    dataset = args.dataset
    bound = args.bound
    multiple = args.multiple
    resolution = res_map[res]

    read_file = os.path.join(data_dir, f"bound{bound}_{res}.txt")
    f = open(read_file, 'w')

    norm_file = os.path.join(data_dir, f"norm.txt")
    fn = open(norm_file, 'w')

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])

    chromosome_expected_size = chromosome_size.get(args.genome_id, {})

    for n in tqdm(chr_list):
        if n in abandon_chromosome:
            continue

        in_file = os.path.join(data_dir, f'chr{n}_{res}.npz')

        matrix, compact_idx, norm = read_npz(in_file, bound = bound, multiple=multiple, include_additional_channels=False)

        if external_norm_file != 'NONE':
            if '#(CHR)' in external_norm_file:
                norm = read_singlechromosome_norm(external_norm_file, n, cell_line, replace_nan = False)
            else:
                raw = norm
        
        fn.write(f'vector\t{norm_name}\tchr{n}\t{resolution} BP\n')

        assert norm.shape[0] == matrix.shape[0]
        size = norm.shape[0]
        if f'chr{n}' in chromosome_expected_size:
            size = min(size, int(np.ceil(chromosome_expected_size[f'chr{n}']/resolution)))

        for i in range(size):
            fn.write(f'{norm[i]}\n')

        
        norm[np.isnan(norm)] = 1
        # print(norm.shape[0])

        for i in tqdm(range(size), leave=False):
            for j in range(max(i-bound,0), min(i+bound, size)):
                RAW = float(matrix[i][j])*norm[i]*norm[j]
                if RAW > 0:
                    f.write(f'{0} chr{n} {i*resolution} {0} {0} chr{n} {j*resolution} {1} {RAW:.2f}\n')

    hic_file = os.path.join(data_dir, f"bound{bound}_{res}.hic")
    os.system(f'~/jdk/jdk1.8.0_391/bin/java -Xmx2g -jar juicer_tools.jar pre -d -n -r {resolution} {read_file} {hic_file} {args.genome_id}')
    
    if norm_name != 'NONE':
        os.system(f'~/jdk/jdk1.8.0_391/bin/java -Xmx2g -jar juicer_tools.jar addNorm {hic_file} {norm_file}')

