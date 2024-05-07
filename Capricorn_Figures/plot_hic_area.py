import pandas as pd
from tqdm import tqdm
import argparse
import os
import numpy as np
from data_processing.Read_npz import read_npz
from HiC_evaluation.bedpe_comparison import read_bedpe_file

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataset_informations import *

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-name', type=str, required=True)

    parser.add_argument('--data-dir', type=str, required = True)
    parser.add_argument('--hic-caption', type=str, default = 'hicarn')
    parser.add_argument('--resolution', type=str, default='10kb')
    parser.add_argument('--chr', type=str, default='17')
    parser.add_argument('--start', type=int, default=4730)
    parser.add_argument('--end', type=int, default=4810)

    parser.add_argument('--plot-loop', action='store_true')
    parser.add_argument('-p', '--predict-bedpe-file', type=str, required=True)
    parser.add_argument('-t', '--target-bedpe-file', type=str, default='/data/hic_data/mat_new/GM12878/10kb/mustache.tsv')

    parser.add_argument('-m', '--matching-scope', type=float, default = 50000)
    parser.add_argument('-n', '--norm-ord', type=float, default = 2)
    parser.add_argument('-d', '--dynamic-scope', action='store_true')

    args = parser.parse_args()

    save_dir = os.path.join('Figures/Fig2')
    os.makedirs(save_dir, exist_ok=True)

    chr = args.chr
    start = args.start
    end = args.end

    matching_scope = args.matching_scope
    norm_ord = args.norm_ord
    dynamic_scope = args.dynamic_scope

    in_file = os.path.join(args.data_dir, f'chr{chr}_{args.resolution}.npz')

    res = res_map[args.resolution]

    pred_matrix, compact_idx, norm = read_npz(in_file, args.hic_caption, include_additional_channels=False)

    file_name = f'{args.method_name}_chr{args.chr}_{start}to{end}.pdf'
    file = os.path.join(save_dir, file_name)
    ax = plt_submatrix(pred_matrix, start, end)

    if args.plot_loop:
        predict_areas = read_bedpe_file(args.predict_bedpe_file)
        target_areas = read_bedpe_file(args.target_bedpe_file)

        pred_areas = []
        tgt_areas = []
        for area in predict_areas[f'chr{chr}']:
            if area[0] >= start*res and area[0] <= end*res and area[1] >= start*res and area[1] <= end*res:
                pred_areas.append(area)
        for area in target_areas[f'chr{chr}']:
            if area[0] >= start*res and area[0] <= end*res and area[1] >= start*res and area[1] <= end*res:
                tgt_areas.append(area)
        
        # print(pred_areas)
        # print(tgt_areas)

        predict_match = np.zeros(len(pred_areas))
        target_match = np.zeros(len(tgt_areas))

        for i, pred_area in enumerate(pred_areas):
            for j, tgt_area in enumerate(tgt_areas):
                pred_idx = np.array(pred_area)
                tgt_idx = np.array(tgt_area)
                
                if dynamic_scope:
                    # dynamic scope when near-diagonal
                    # More strict about near-diagonal areas
                    scope = min(matching_scope, 0.2 * abs(pred_idx[0]-pred_idx[1]))
                else:
                    scope = matching_scope

                if np.linalg.norm(pred_idx - tgt_idx, norm_ord) <= scope:
                    predict_match[i] = 1
                    target_match[j] = 1

        matched_peak_top = []
        unmatched_peak_top = []
        for i, match in enumerate(predict_match):
            if match == 1:
                matched_peak_top.append(pred_areas[i])
            else:
                unmatched_peak_top.append(pred_areas[i])

        if len(matched_peak_top)>0:
            peak_top = np.array(matched_peak_top)//res - start
            ax.scatter(peak_top[:, 1], peak_top[:, 0], s = 2, linewidths = 0.2, facecolors='blue', edgecolors='blue')

        if len(unmatched_peak_top)>0:
            peak_top = np.array(unmatched_peak_top)//res - start
            ax.scatter(peak_top[:, 1], peak_top[:, 0], s = 2, linewidths = 0.2, facecolors='none', edgecolors='blue')
    
    plt.savefig(file, dpi=end-start)
