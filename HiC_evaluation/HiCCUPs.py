# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# our implementation of HiCCUPs algorithm.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
from HiC_evaluation.utils import *
from HiC_evaluation.dataset_info import *
from HiC_evaluation.args import *

from scipy.ndimage import correlate
from scipy.stats import poisson

import warnings
warnings.filterwarnings("ignore")
import logging

def donut_kernel(args):
    R1 = args.donut_size
    R2 = args.peak_size

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[center[0], :] = 0
    kernel[:, center[1]] = 0

    return kernel

def lowerleft_kernel(args):
    R1 = args.donut_size
    R2 = args.peak_size

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[:center[0]+1, :] = 0
    kernel[:, center[1]:] = 0

    return kernel

def horizontal_kernel(args):
    R1 = args.donut_size
    R2 = args.peak_size
    
    kernel = np.zeros((3, R1*2+1))
    center = (1, R1)

    kernel[ : , : center[1] - R2 ] = 1
    kernel[ : , center[1] + R2 + 1 : ] = 1
    
    return kernel

def vertical_kernel(args):
    R1 = args.donut_size
    R2 = args.peak_size
    
    kernel = np.zeros((R1*2+1, 3))
    center = (R1, 1)

    kernel[ : center[0] - R2, : ] = 1
    kernel[ center[0] + R2 + 1 : , : ] = 1

    return kernel

def loop_clustering(args, peak_cands, info = True):
    num_cands = len(peak_cands)
    peaks_final = []
    while len(peak_cands) > 0:
        top_peak = max(peak_cands)
        peak_cands.remove(top_peak)
        peaks_cluster = [top_peak[1]]
        centroid = top_peak[1]
        r = 0
        find = True
        while find:
            find = False

            def dis(x, y):
                return np.linalg.norm((x[0]-y[0], x[1]-y[1]))
            
            centroid = np.mean(peaks_cluster, axis = 0)
            r = max([dis(peak, centroid) for peak in peaks_cluster ])
                
            for peak in peak_cands:
                if dis(peak[1], centroid) - r < args.clustering_boundary:
                    peaks_cluster.append(peak[1])
                    peak_cands.remove(peak)
                    find = True
                    break
        if r>0 or top_peak[2] <= args.singleton_qvalue:
            peaks_final.append((top_peak[1], centroid, r))
    
    if info:
        print_info(f'Found {len(peaks_final)} peaks from {num_cands} candidates')

    return peaks_final

def find_peaks(full_matrix, full_norm, compact_idx, args, return_qvalue = False, info = True):

    kernels = [donut_kernel(args), lowerleft_kernel(args), horizontal_kernel(args), vertical_kernel(args)]
    l = full_matrix.shape[0]
    B = min(args.bounding, l)
    window_size = min(2*B, l)
    upper_triangle = np.triu(np.ones((window_size, window_size)), 0)
    
    expect_vector = get_oe_matrix(full_matrix, bounding = window_size, oe=False)

    expect = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            if abs(i-j) < len(expect_vector):
                expect[i][j] = expect_vector[abs(i-j)]
    esums = []
    for kernel in kernels:
        esum = correlate(expect, kernel, mode='constant')
        esums.append(esum)

    enriched_pixels = []

    first_patch_qvalues = np.copy(upper_triangle)
    if info:
        pbar = tqdm(range(0, l, B))
    else:
        pbar = range(0, l, B)
    for s0 in pbar:
        if info:
            pbar.set_description(f'Currently find {len(enriched_pixels)} enriched pixels')
            
        s = min(s0, l-window_size)
        matrix = full_matrix[s:s+window_size, s:s+window_size]
        norm   = full_norm  [s:s+window_size]

        matrix = matrix * args.multiple 
        norm_mat = np.outer(norm, norm)

        observed = matrix * norm_mat

        observed = (np.rint(observed)).astype(int)

        log_lambda_step = np.log(args.lambda_step)

        pixel_scores = {}

        # print(observed)

        for kid, kernel in enumerate(kernels):
            msum = correlate(matrix, kernel, mode='constant')
            esum = esums[kid]

            Ek = np.nan_to_num(msum/esum*expect)
            Ek = Ek * norm_mat
            
            #lambda-chunk FDR

            logEk = np.nan_to_num(np.log(Ek))

            bin_id = np.maximum(0, np.ceil(logEk/log_lambda_step).astype(int))
            pvalues = poisson.sf(observed, np.exp(bin_id*log_lambda_step))

            max_bin = bin_id.max()+1       
            
            for id in range(max_bin):
                bin_pos = np.where((bin_id == id) & (upper_triangle == 1))
                p = pvalues[bin_pos]

                bin = sorted(zip(p.tolist(), bin_pos[0].tolist(), bin_pos[1].tolist()))
                size = len(bin)

                qvalue = 1
                for rank in range(len(bin), 0, -1):
                    pvalue, i, j = bin[rank-1]
                    qvalue = min(qvalue, pvalue /(rank / size))

                    if s==0:
                        first_patch_qvalues[i,j] = min(first_patch_qvalues[i,j], 1-qvalue if qvalue <= args.FDR else 0)

                    if qvalue <= args.FDR and observed[i][j]/Ek[i][j] > args.thresholds[kid]:
                        # pass BHFDR, check ratio
                        flag = (kid in (0,1)) and (observed[i][j]/Ek[i][j] > args.thresholds[-1])
                        if (i,j) not in pixel_scores: pixel_scores[(i,j)] = [0, 0]
                        pixel_scores[(i,j)][0] += 2 + (1 if flag else 0)
                        pixel_scores[(i,j)][1] += qvalue
                
        for p, v in pixel_scores.items():
            if v[0]>=9 and abs(p[0]-p[1]) <= args.bounding:
                enriched_pixels.append((observed[p[0], p[1]], (p[0]+s, p[1]+s), v[1]))
    
    gaps = set(range(l)) - set(compact_idx)
    near_gap = [False for _ in range(l)]
    for gap in gaps:
        for i in range(args.gap_filter_range):
            if gap-i >= 0: 
                near_gap[gap-i] = True
            if gap+i < l:
                near_gap[gap+i] = True
    
    filtered_enriched_pixels = []
    for pixels in enriched_pixels:
        if not near_gap[pixels[1][0]] and not near_gap[pixels[1][1]]:
            filtered_enriched_pixels.append(pixels)

    peaks_final = loop_clustering(args, filtered_enriched_pixels, info=info)

    if return_qvalue:
        return peaks_final, first_patch_qvalues
    else:
        return peaks_final

def find_peaks_with_qvalue_and_ratio(full_matrix, full_norm, qvalues, ratios, args, compact_idx = None, return_qvalue=False, info=True):
    l = full_matrix.shape[0]

    for i in range(qvalues.shape[-1]):
        if i not in compact_idx:
            qvalues[i, :] = 1
            qvalues[:, i] = 1

    enriched_pixels_pos = np.where(np.logical_and(qvalues<=args.FDR, ratios>=args.ratio_channel_threshold))
    enriched_pixels = []
    for x,y in zip(enriched_pixels_pos[0], enriched_pixels_pos[1]):
        if x<=y and abs(x-y) <= args.bounding:
            enriched_pixels.append( (full_matrix[x,y] * full_norm[x] * full_norm[y], (x, y), qvalues[x, y] ))

    gaps = set(range(l)) - set(compact_idx)
    near_gap = [False for _ in range(l)]
    for gap in gaps:
        for i in range(args.gap_filter_range):
            if gap-i >= 0: 
                near_gap[gap-i] = True
            if gap+i < l:
                near_gap[gap+i] = True
    
    filtered_enriched_pixels = []
    for pixels in enriched_pixels:
        if not near_gap[pixels[1][0]] and not near_gap[pixels[1][1]]:
            filtered_enriched_pixels.append(pixels)

    peaks_final = loop_clustering(args, filtered_enriched_pixels, info=info)

    if return_qvalue:
        return peaks_final, np.triu((1 - qvalues) * (qvalues<=args.FDR), 0)
    else:
        return peaks_final

def peak_similarity(pred_peaks, tgt_peaks, max_scope=5.0, info=True):
    # compute the f1 score of two peak sets
    predicted = len(pred_peaks)
    target = len(tgt_peaks)
    if target <= 0: return -1
    matched = 0
    for pred_peak in pred_peaks:
        pred_idx = np.array(pred_peak[0])
        for tgt_peak in tgt_peaks:
            tgt_idx = np.array(tgt_peak[0])
            scope = min(max_scope, 0.2 * abs(pred_idx[0]-pred_idx[1]))
            if np.linalg.norm(pred_idx - tgt_idx) <= scope:
                matched += 1
                break
    
    if info:
        print_info(f'matched: {matched}, predicted: {predicted}, target: {target}')
    
    precision = matched / predicted if predicted else 0
    recall = matched / target if target else 0

    f1 = 2 * matched / (predicted + target) #equivalent to 2/(1/precision+1/recall)
    return f1, matched

import matplotlib.pyplot as plt

def draw_peaks(matrix, peaks, start, end, file_name):
    plt.clf()
    plt.imshow(matrix[start:end, start:end], cmap='OrRd')
    plt.colorbar()
    peak_top = []
    peak_center = []
    peak_size = []
    for p, c, r in peaks:
        if start <= p[0] and p[0] < end and start <= p[1] and p[1] < end:
            peak_top.append((p[0]-start, p[1]-start))
            peak_center.append((c[0]-start, c[1]-start))
            peak_size.append(r+20)

    if len(peak_top)>0:
        peak_top = np.array(peak_top)
        peak_center = np.array(peak_center)
        # plt.scatter(peak_top[:, 1], peak_top[:, 0], s = 5, linewidths = 0.5, facecolors='none', edgecolors='red')
        plt.scatter(peak_center[:, 1], peak_center[:, 0], s = peak_size, linewidths = 0.5, facecolors='none', edgecolors='blue')
    
    plt.savefig(file_name)

if __name__ == '__main__':
    parser = evaluate_parser()
    parser.add_argument('--norm-file', type=str, default = '/data/hic_data/raw/#DATASET/10kb_resolution_intrachromosomal/#CHR/MAPQGE30/#CHR_10kb.KRnorm', help='The path to store norm file. #CHR and #DATASET will automaticly be replaced with the corresponding values.')
    parser.add_argument('--multiple', type=int, default=255, help='The cutoff used to generate data. Multiply this to the matrices before trying to find loops')

    parser.add_argument('--peak-size', type=int, default = 2)
    parser.add_argument('--donut-size', type=int, default = 5)
    parser.add_argument('--lambda-step', type=float, default=2**(1/3))
    parser.add_argument('--FDR', type=float, default=0.1)
    parser.add_argument('--clustering-boundary', type=float, default=2)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1.75, 1.75, 1.5, 1.5, 2])
    parser.add_argument('--gap-filter-range', type=int, default=5)
    parser.add_argument('--singleton-qvalue', type=float, default=0.02)
    
    parser.add_argument('--use-extra-channels', action='store_true')
    parser.add_argument('--qvalue-channel', type=int, default=3)
    parser.add_argument('--ratio-channel', type=int, default=4)
    parser.add_argument('--ratio-channel-threshold', type=float, default = 0.8)

    args = parser.parse_args()
    chr_list = dataset_chrs[args.dataset]

    f1_scores = []
    matched_counts = []
    predict_loop_counts = []

    extrachannel_f1_scores = []
    extrachannel_matched_counts = []
    extrachannel_predict_loop_counts = []

    target_loop_counts = []
    
    save_dir = os.path.join(args.predict_dir, 'HiCCUPs' if args.save_name is None else args.save_name)
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    for chr in chr_list:
        full_pred_matrix, full_target_matrix, compact_idx = read_matrices(args, chr, compact_idx = True)

        if len(full_pred_matrix.shape)>=3 :
            pred_matrix = full_pred_matrix[0]
        else:
            pred_matrix = full_pred_matrix
        if len(full_target_matrix.shape)>=3 :
            target_matrix = full_target_matrix[0]
        else:
            target_matrix = full_target_matrix

        
        normFile = args.norm_file
        norm_dataset = args.dataset[:-9] if args.dataset[-9:] == '_crosschr' else args.dataset
        norm_dataset = 'CH12-LX' if norm_dataset == 'mESC' else norm_dataset
        normFile = normFile.replace('#DATASET', norm_dataset)
        normFile = normFile.replace('#CHR', 'chr'+str(chr))

        norm = open(normFile, 'r').readlines()
        norm = np.array(list(map(float, norm)))
        norm[np.isnan(norm)] = 1

        tgt_peak, tgt_qvalue = find_peaks(target_matrix, norm, compact_idx, args, return_qvalue=True)
        target_loop_counts.append(len(tgt_peak))
        # draw_peaks(target_matrix, tgt_peak, start = 100, end = 140, file_name=os.path.join(save_dir, f'chr{chr}_tgt.pdf'))
        # draw_peaks(tgt_qvalue, tgt_peak, start = 100, end = 140, file_name=os.path.join(save_dir, f'chr{chr}_tgt_qvalue.pdf'))

        pred_peak, pred_qvalue = find_peaks(pred_matrix, norm, compact_idx, args, return_qvalue=True)
        predict_loop_counts.append(len(pred_peak))
        # draw_peaks(pred_matrix, pred_peak, start = 100, end = 140, file_name=os.path.join(save_dir, f'chr{chr}_pred.pdf'))
        # draw_peaks(pred_qvalue, pred_peak, start = 100, end = 140, file_name=os.path.join(save_dir, f'chr{chr}_pred_qvalue.pdf'))
        # draw_peaks(pred_qvalue, pred_peak, start = 100, end = 140, file_name=os.path.join(save_dir, f'chr{chr}_pred_qvalue.pdf'))

        f1_score, matched_count = peak_similarity(pred_peak, tgt_peak)
        print_info(f"loop f1 score for chr {chr} : {f1_score:.4f}")
        f1_scores.append(f1_score)
        matched_counts.append(matched_count)

        if args.use_extra_channels:
            pred_peak_1, pred_qvalue_1 = find_peaks_with_qvalue_and_ratio(pred_matrix, norm, full_pred_matrix[args.qvalue_channel], full_pred_matrix[args.ratio_channel], args, compact_idx, return_qvalue=True)
            extrachannel_predict_loop_counts.append(len(pred_peak_1))
            # draw_peaks(pred_matrix, pred_peak_1, start = 100, end = 300, file_name=os.path.join(save_dir, f'chr{chr}_pred_1.pdf'))
            # draw_peaks(pred_qvalue_1, pred_peak_1, start = 100, end = 300, file_name=os.path.join(save_dir, f'chr{chr}_pred_qvalue_1.pdf'))

            f1_score, matched_count = peak_similarity(pred_peak_1, tgt_peak)
            print_info(f"[with extra channels]loop f1 score with extra channel for chr {chr} : {f1_score:.4f}")
            extrachannel_f1_scores.append(f1_score)
            extrachannel_matched_counts.append(matched_count)

    print_info(f'target loop counts:{target_loop_counts}')
    print_info(f'predict loop counts:{predict_loop_counts}')
    print_info(f'matched loop counts:{matched_counts}')

    print_info('chromosome loop f1 score:')
    print_info(f"{f1_scores}")
    print_info(f"chromosome-level average loop f1 score:{np.mean(f1_scores):.4f}")

    if args.use_extra_channels:
        print_info(f'[with extra channels]predict loop counts:{extrachannel_predict_loop_counts}')
        print_info(f'[with extra channels]matched loop counts:{matched_counts}')

        print_info('[with extra channels]chromosome loop f1 score:')
        print_info(f"{extrachannel_f1_scores}")
        print_info(f"[with extra channels]chromosome-level average loop f1 score:{np.mean(extrachannel_f1_scores):.4f}")
