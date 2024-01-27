# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Run HiCCUPS on every submatrix and report the statistics on each submatrix. 
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
from HiC_evaluation.utils import *
from dataset_informations import *
from HiC_evaluation.HiCCUPS import *
from HiC_evaluation.bedpe_comparison import area_similarity

import warnings
warnings.filterwarnings("ignore")
import logging

import json

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict-dir', type=str, required=True)
    parser.add_argument('-pr', '--predict-resolution', type=str, default='10kb')
    parser.add_argument('-pc', '--predict-caption', type=str, default='hic')
    parser.add_argument('-pm', '--predict-multiple', type=int, default=255)
    parser.add_argument('--predict-external-norm-file', type=str, 
                        default = '/data/hic_data/raw/#(CELLLINE)/10kb_resolution_intrachromosomal/#(CHR)/MAPQGE30/#(CHR)_10kb.KRnorm')

    parser.add_argument('-t', '--target-dir', type=str, default='/data/hic_data/mat_new/GM12878')
    parser.add_argument('-tr', '--target-resolution', type=str, default='10kb')
    parser.add_argument('-tc', '--target-caption', type=str, default='hic')
    parser.add_argument('-tm', '--target-multiple', type=int, default=1)
    parser.add_argument('--target-external-norm-file', type=str, 
                        default = '/data/hic_data/raw/#(CELLLINE)/10kb_resolution_intrachromosomal/#(CHR)/MAPQGE30/#(CHR)_10kb.KRnorm')

    parser.add_argument('--bound', type=int, default=200)

    parser.add_argument('-c', '--cell-line', default='GM12878')
    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('--peak-size', type=int, default = 2)
    parser.add_argument('--min-donut-size', type=int, default = 5)
    parser.add_argument('--max-donut-size', type=int, default = 5)
    parser.add_argument('--min-reads', type=int, default = 16)
    parser.add_argument('--lambda-step', type=float, default=2**(1/3))
    parser.add_argument('--FDR', type=float, default=0.1)
    parser.add_argument('--clustering-boundary', type=float, default=2)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1.75, 1.75, 1.5, 1.5, 2])
    parser.add_argument('--gap-filter-range', type=int, default=5)
    parser.add_argument('--singleton-qvalue', type=float, default=0.02)

    parser.add_argument('-m', '--matching-scope', type=float, default = 50000)
    parser.add_argument('-n', '--norm-ord', type=float, default = 2)
    parser.add_argument('-d', '--dynamic-scope', action='store_true')
    
    parser.add_argument('--stride', type=int, default=400)
    parser.add_argument('--size', type=int, default=400)

    parser.add_argument('-mn', '--method-name', type=str, required=True)
    args = parser.parse_args()
    
    bound = args.bound

    cell_line = args.cell_line
    dataset = args.dataset
    
    peak_size = args.peak_size
    min_donut_size = args.min_donut_size
    max_donut_size = args.max_donut_size
    min_reads = args.min_reads
    lambda_step = args.lambda_step
    FDR = args.FDR
    thresholds = args.thresholds
    gap_filter_range = args.gap_filter_range
    clustering_boundary = args.clustering_boundary
    singleton_qvalue = args.singleton_qvalue

    matching_scope = args.matching_scope
    norm_ord = args.norm_ord
    dynamic_scope = args.dynamic_scope

    if norm_ord >= 10:
        norm_ord = np.inf
    elif norm_ord <= -10:
        norm_ord = -np.inf

    chr_list = set_dict[args.dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])

    save_dir = os.path.join('results/submatrices_results', args.method_name, cell_line)
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    results = []

    for n in tqdm(chr_list):
        if n in abandon_chromosome:
            continue

        pred_file = os.path.join(args.predict_dir, f'chr{n}_{args.predict_resolution}.npz')
        
        pred_matrix, pred_compact, pred_norm = read_npz(pred_file, 
                                                        hic_caption = args.predict_caption, 
                                                        multiple = args.predict_multiple,
                                                        bound = bound, 
                                                        include_additional_channels=False
                                                        )
        
        if args.predict_external_norm_file != 'NONE':
            if '#(CHR)' in args.predict_external_norm_file:
                pred_norm = read_singlechromosome_norm(args.predict_external_norm_file, n, cell_line)
            else:
                raise NotImplementedError
        
        target_file = os.path.join(args.target_dir, f'chr{n}_{args.target_resolution}.npz')
        target_matrix, target_compact, target_norm = read_npz(target_file, 
                                                              hic_caption = args.target_caption,
                                                              multiple = args.target_multiple,
                                                              bound = bound, 
                                                              include_additional_channels=False
                                                              )
        if args.target_external_norm_file != 'NONE':
            if '#(CHR)' in args.target_external_norm_file:
                pred_norm = read_singlechromosome_norm(args.target_external_norm_file, n, cell_line)
            else:
                raise NotImplementedError

        for start in tqdm(range(0, target_matrix.shape[-1]-args.size, args.stride), leave=False):
            end = start + args.size

            ptarget_matrix = target_matrix[start:end, start:end]
            ptarget_norm = target_norm[start:end]
            ptarget_compact = [c-start for c in target_compact if c >= start and c < end]

            ppred_matrix = pred_matrix[start:end, start:end]
            ppred_norm = pred_norm[start:end]
            ppred_compact = [c-start for c in pred_compact if c >= start and c < end]

            if len(ptarget_compact) < 300: continue

            target_peaks = find_peaks(
                            ptarget_matrix, 
                            ptarget_norm, 
                            ptarget_compact, 
                            bound,
                            peak_size,
                            min_donut_size,
                            max_donut_size,
                            min_reads,
                            lambda_step,
                            FDR,
                            thresholds,
                            gap_filter_range,
                            clustering_boundary,
                            singleton_qvalue,
                            info=False
                            )
            
            target_peaks = [peak[0] for peak in target_peaks]
            
            if len(target_peaks) < 1: continue
            pred_peaks = find_peaks(
                            ppred_matrix, 
                            ppred_norm, 
                            ppred_compact, 
                            bound,
                            peak_size,
                            min_donut_size,
                            max_donut_size,
                            min_reads,
                            lambda_step,
                            FDR,
                            thresholds,
                            gap_filter_range,
                            clustering_boundary,
                            singleton_qvalue,
                            info=False
                            )
            
            pred_peaks = [peak[0] for peak in pred_peaks]
            
            pred_matched, target_matched, precision, recall, f1 = area_similarity(pred_peaks, target_peaks, 
                                             matching_scope=5, 
                                             norm_ord=norm_ord,  
                                             dynamic_scope=dynamic_scope,
                                             info=False)

            results.append((n, start, len(pred_peaks), len(target_peaks), pred_matched, target_matched, precision, recall, f1))

    chrs, pos, num_pred, num_target, pred_matched, target_matched, precisions, recalls, f1s = list(map(list, zip(*results)))
    result_data = pd.DataFrame.from_dict({
        'chromosome' : chrs,
        'position' : pos,
        'predicted' : num_pred,
        'target' : num_target,
        'matched predicted' : pred_matched,
        'matched target' : target_matched,
        'precision' : precisions,
        'recall' : recalls,
        'F1 score': f1s
        })

    result_file = os.path.join(save_dir, 'results.tsv')
    result_data.to_csv(result_file, sep='\t', index = False)


