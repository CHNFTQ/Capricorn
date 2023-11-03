import pandas as pd
from tqdm import tqdm
import argparse
import os
import numpy as np
from HiC_evaluation.utils import *
from HiC_evaluation.dataset_info import *
from HiC_evaluation.args import *
from HiC_evaluation.HiCCUPs import find_peaks

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RED = np.array((1, 0, 0))
WHITE = np.array((1, 1, 1))
    
colors = [WHITE + t * (RED - WHITE) for t in np.arange(0, 1, 1e-3)]
cmap = ListedColormap(colors)

def plt_submatrix(matrix, start, end):
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(matrix[start:end, start:end], cmap=cmap)

    return ax

if __name__ == '__main__':
    parser = evaluate_parser()
    parser.add_argument('--method-name', type=str, default = 'HR')
    parser.add_argument('--chr', type=int, default=17)
    parser.add_argument('--start', type=int, default=4730)
    parser.add_argument('--end', type=int, default=4810)
    parser.add_argument('--plot-loop', action='store_true')

    parser.add_argument('--norm-file', type=str, default = '/data/hic_data/raw/#DATASET/10kb_resolution_intrachromosomal/#CHR/MAPQGE30/#CHR_10kb.KRnorm')
    parser.add_argument('--peak-size', type=int, default = 2)
    parser.add_argument('--donut-size', type=int, default = 5)
    parser.add_argument('--lambda-step', type=float, default=2**(1/3))
    parser.add_argument('--FDR', type=float, default=0.1)
    parser.add_argument('--clustering-boundary', type=float, default=2)
    parser.add_argument('--gap-filter-range', type=int, default=5)
    parser.add_argument('--multiple', type=int, default=255)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1.75, 1.75, 1.5, 1.5, 2])
    parser.add_argument('--singleton-qvalue', type=float, default=0.02)

    args = parser.parse_args()

    save_dir = os.path.join('Figures/Fig2')
    os.makedirs(save_dir, exist_ok=True)

    start = args.start
    end = args.end

    
    pred_matrix, target_matrix, compact_idx = read_matrices(args, args.chr, compact_idx=True)

    if len(pred_matrix.shape)>=3 :
        pred_matrix = pred_matrix[0]
    else:
        pred_matrix = pred_matrix

    file_name = f'{args.method_name}_chr{args.chr}_{args.start}to{args.end}.pdf'
    file = os.path.join(save_dir, file_name)
    ax = plt_submatrix(pred_matrix, args.start, args.end)

    peaks = None
    if args.plot_loop:
        normFile = args.norm_file
        normFile = normFile.replace('#DATASET', 'CH12-LX' if args.dataset == 'mESC' else args.dataset)
        normFile = normFile.replace('#CHR', 'chr'+str(args.chr))

        norm = open(normFile, 'r').readlines()
        norm = np.array(list(map(float, norm)))
        norm[np.isnan(norm)] = 1

        ppred_matrix = pred_matrix[..., start:end, start:end]
        ptarget_matrix = target_matrix[..., start:end, start:end]
        pnorm = norm[start:end]

        pcompact = [c-start for c in compact_idx if c >= start and c < end]
        pred_peaks = find_peaks(ppred_matrix, pnorm, pcompact, args, return_qvalue=False, info=False)
        target_peaks = find_peaks(ptarget_matrix, pnorm, pcompact, args, return_qvalue=False, info=False)

        matched_peak_top = []
        unmatched_peak_top = []
        for pred_peak in pred_peaks:
            pred_idx = np.array(pred_peak[0])
            flag = False
            for tgt_peak in target_peaks:
                tgt_idx = np.array(tgt_peak[0])
                scope = min(5, 0.2 * abs(pred_idx[0]-pred_idx[1]))
                if np.linalg.norm(pred_idx - tgt_idx) <= scope:
                    flag = True
                    break
            if flag:
                matched_peak_top.append(pred_peak[0])
            else:
                unmatched_peak_top.append(pred_peak[0])

        if len(matched_peak_top)>0:
            peak_top = np.array(matched_peak_top)
            ax.scatter(peak_top[:, 1], peak_top[:, 0], s = 2, linewidths = 0.2, facecolors='blue', edgecolors='blue')

        if len(unmatched_peak_top)>0:
            peak_top = np.array(unmatched_peak_top)
            ax.scatter(peak_top[:, 1], peak_top[:, 0], s = 2, linewidths = 0.2, facecolors='none', edgecolors='blue')
    
    plt.savefig(file, dpi=end-start)
