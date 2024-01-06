# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# our implementation of HiCCUPs algorithm.
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import os
import argparse
from HiC_evaluation.utils import *
from data_processing.Read_npz import read_npz
from dataset_informations import *
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from scipy.stats import poisson
import json

import warnings
warnings.filterwarnings("ignore")
import logging

def donut_kernel(R2, R1):

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[center[0], :] = 0
    kernel[:, center[1]] = 0

    return kernel

def lowerleft_kernel(R2, R1):

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[:center[0]+1, :] = 0
    kernel[:, center[1]:] = 0

    return kernel

def horizontal_kernel(R2, R1):

    kernel = np.zeros((3, R1*2+1))
    center = (1, R1)

    kernel[ : , : center[1] - R2 ] = 1
    kernel[ : , center[1] + R2 + 1 : ] = 1
    
    return kernel

def vertical_kernel(R2, R1):

    kernel = np.zeros((R1*2+1, 3))
    center = (R1, 1)

    kernel[ : center[0] - R2, : ] = 1
    kernel[ center[0] + R2 + 1 : , : ] = 1

    return kernel

def get_kernels(peak_size, donut_size):
    return [donut_kernel(peak_size, donut_size), 
            lowerleft_kernel(peak_size, donut_size), 
            horizontal_kernel(peak_size, donut_size), 
            vertical_kernel(peak_size, donut_size)]

def loop_clustering(peak_cands, 
                    clustering_boundary,
                    singleton_qvalue,
                    info = True):
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
                if dis(peak[1], centroid) - r < clustering_boundary:
                    peaks_cluster.append(peak[1])
                    peak_cands.remove(peak)
                    find = True
                    break
        if r>0 or top_peak[2] <= singleton_qvalue:
            peaks_final.append((top_peak[1], centroid, r))
    
    if info:
        print_info(f'Found {len(peaks_final)} peaks from {num_cands} candidates')

    return peaks_final

def find_peaks(full_matrix,
               full_norm, 
               compact_idx, 
               bound, 

               peak_size,
               donut_size,
               lambda_step,
               FDR,
               thresholds,
               gap_filter_range,
               clustering_boundary,
               singleton_qvalue,

               info = True):

    kernels = get_kernels(peak_size, donut_size)

    l = full_matrix.shape[0]
    B = min(bound, l)
    window_size = min(2*B, l)
    upper_triangle = np.triu(np.ones((window_size, window_size)), 0)
    
    expect_vector = get_oe_matrix(full_matrix, bound = window_size, oe=False)

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

        norm_mat = np.outer(norm, norm)

        observed = matrix * norm_mat

        observed = (np.rint(observed)).astype(int)

        log_lambda_step = np.log(lambda_step)

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

                    if qvalue <= FDR and observed[i][j]/Ek[i][j] > thresholds[kid]:
                        # pass BHFDR, check ratio
                        flag = (kid in (0,1)) and (observed[i][j]/Ek[i][j] > thresholds[-1])
                        if (i,j) not in pixel_scores: pixel_scores[(i,j)] = [0, 0]
                        pixel_scores[(i,j)][0] += 2 + (1 if flag else 0)
                        pixel_scores[(i,j)][1] += qvalue
                
        for p, v in pixel_scores.items():
            if v[0]>=9 and abs(p[0]-p[1]) <= bound:
                enriched_pixels.append((observed[p[0], p[1]], (p[0]+s, p[1]+s), v[1]))
    
    gaps = set(range(l)) - set(compact_idx)
    near_gap = [False for _ in range(l)]
    for gap in gaps:
        for i in range(gap_filter_range):
            if gap-i >= 0: 
                near_gap[gap-i] = True
            if gap+i < l:
                near_gap[gap+i] = True
    
    filtered_enriched_pixels = []
    for pixels in enriched_pixels:
        if not near_gap[pixels[1][0]] and not near_gap[pixels[1][1]] and abs(pixels[1][0] - pixels[1][1]) > peak_size + 2:
            filtered_enriched_pixels.append(pixels)

    peaks_final = loop_clustering(filtered_enriched_pixels, clustering_boundary, singleton_qvalue, info=info)

    return peaks_final

def draw_peaks(matrix, peaks, start, end, file_name):
    plt.clf()
    plt.imshow(matrix[start:end, start:end], vmax=255, cmap='OrRd')
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
        # peak_center = np.array(peak_center)
        plt.scatter(peak_top[:, 1], peak_top[:, 0], s = 5, linewidths = 0.5, facecolors='none', edgecolors='red')
        # plt.scatter(peak_center[:, 1], peak_center[:, 0], s = peak_size, linewidths = 0.5, facecolors='none', edgecolors='blue')
    
    plt.savefig(file_name)

def write_bedpe_annotation(chr, peaks, file, resolution):
    for top, center, radius in peaks:
        x1 = int(top[0] * resolution)
        x2 = int(top[0] * resolution + resolution)
        y1 = int(top[1] * resolution)
        y2 = int(top[1] * resolution + resolution)
        file.write(f'{chr}\t{x1}\t{x2}\t{chr}\t{y1}\t{y2}\t.\t.\t.\t.\t0,0,255\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, required = True)
    parser.add_argument('--hic-caption', type=str, default = 'hic')
    parser.add_argument('--external-norm-file', type=str, 
                        default = '/data/hic_data/raw/GM12878/10kb_resolution_intrachromosomal/#(CHR)/MAPQGE30/#(CHR)_10kb.KRnorm')
    parser.add_argument('--resolution', type=str, default='10kb')

    parser.add_argument('--bound', type=int, default=200)
    parser.add_argument('--multiple', type=int, default=255)

    parser.add_argument('-c', '--cell-line', default='GM12878')
    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('--peak-size', type=int, default = 2)
    parser.add_argument('--donut-size', type=int, default = 5)
    parser.add_argument('--lambda-step', type=float, default=2**(1/3))
    parser.add_argument('--FDR', type=float, default=0.1)
    parser.add_argument('--clustering-boundary', type=float, default=2)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1.75, 1.75, 1.5, 1.5, 2])
    parser.add_argument('--gap-filter-range', type=int, default=5)
    parser.add_argument('--singleton-qvalue', type=float, default=0.02)

    args = parser.parse_args()

    data_dir = args.data_dir
    hic_caption = args.hic_caption
    external_norm_file = args.external_norm_file
    res = args.resolution
    bound = args.bound
    multiple = args.multiple

    cell_line = args.cell_line
    dataset = args.dataset
    
    resolution = res_map[res]

    peak_size = args.peak_size
    donut_size = args.donut_size
    lambda_step = args.lambda_step
    FDR = args.FDR
    thresholds = args.thresholds
    gap_filter_range = args.gap_filter_range
    clustering_boundary = args.clustering_boundary
    singleton_qvalue = args.singleton_qvalue

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])

    save_dir = os.path.join(data_dir, res, 'HiCCUPS')
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

    loop_counts = []
    annotation = open(os.path.join(save_dir, 'HiCCUPS_loop_annotation.bedpe'), 'w')
    annotation.write(f'chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tname\tscore\tstrand1\tstrand2\tcolor\n')
    for n in chr_list:
        if n in abandon_chromosome:
            continue

        in_file = os.path.join(data_dir, f'chr{n}_{res}.npz')
        
        matrix, compact_idx, norm = read_npz(in_file, hic_caption=hic_caption, bound = bound, multiple=multiple, include_additional_channels=False)

        if external_norm_file is not None:
            if '#(CHR)' in external_norm_file:
                CHR_norm_File = external_norm_file
                CHR_norm_File = CHR_norm_File.replace('#(CHR)', 'chr'+str(n))

                norm = open(CHR_norm_File, 'r').readlines()
                norm = np.array(list(map(float, norm)))
                norm[np.isnan(norm)] = 1
            else:
                raise NotImplementedError

        peaks = find_peaks(matrix, 
                           norm, 
                           compact_idx, 
                           bound,
                           peak_size,
                           donut_size,
                           lambda_step,
                           FDR,
                           thresholds,
                           gap_filter_range,
                           clustering_boundary,
                           singleton_qvalue)

        start = 0
        end = 400                           
        draw_peaks(matrix, peaks, start = start, end = end, file_name=os.path.join(save_dir, f'chr{n}_{start}to{end}.pdf'))

        loop_counts.append(len(peaks))
        write_bedpe_annotation('chr'+str(n), peaks, annotation, resolution)

    print_info(f'Chromosome loop counts: {loop_counts}')
    print_info(f'Total loop counts: {np.sum(loop_counts)}')
