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
from HiC_evaluation.HiCCUPs import *

import warnings
warnings.filterwarnings("ignore")
import logging

import json

import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd

def using_datashader(ax, x, y):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        norm="log",
        aspect="auto",
        ax=ax,
    )

    plt.colorbar(dsartist)

def using_hist2d(ax, x, y, range = None, bins=(100, 100)):
    # https://stackoverflow.com/a/20105673/3015186
    # Answer by askewchan
    _, _, _, im = ax.hist2d(x, y, bins, range=range, cmap=plt.cm.jet)
    plt.colorbar(im)


from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x = [x[i] for i in idx]
        y = [y[i] for i in idx]
        z = [z[i] for i in idx]
        
    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


if __name__ == '__main__':
    parser = evaluate_parser()
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

    parser.add_argument('--stride', type=int, default=400)
    parser.add_argument('--size', type=int, default=400)
    parser.add_argument('--method-titles', nargs='+', type=str, default=['groundtruth', 'HiCARN-2'])

    args = parser.parse_args()
    chr_list = dataset_chrs[args.dataset]

    pred_results = []

    save_dir = os.path.join('Figures/Fig3', args.method_titles[-1], args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    for chr in tqdm(chr_list):
        if chr == 'X': continue
        full_pred_matrix, full_target_matrix, compact_idx = read_matrices(args, chr, compact_idx = True)
       
        if len(full_target_matrix.shape)>=3 :
            target_matrix = full_target_matrix[0]
        else:
            target_matrix = full_target_matrix

        if len(full_pred_matrix.shape)>=3 :
            pred_matrix = full_pred_matrix[0]
        else:
            pred_matrix = full_pred_matrix
        
        normFile = args.norm_file
        normFile = normFile.replace('#DATASET', 'CH12-LX' if args.dataset == 'mESC' else args.dataset)
        normFile = normFile.replace('#CHR', 'chr'+str(chr))

        norm = open(normFile, 'r').readlines()
        norm = np.array(list(map(float, norm)))
        norm[np.isnan(norm)] = 1

        for start in tqdm(range(0, target_matrix.shape[-1]-args.size, args.stride), leave=False):
            end = start + args.size

            ptarget_matrix = target_matrix[..., start:end, start:end]
            ppred_matrix = pred_matrix[..., start:end, start:end]
            pnorm = norm[start:end]

            pcompact = [c-start for c in compact_idx if c >= start and c < end]

            if len(pcompact) < 300: continue

            target_peak = find_peaks(ptarget_matrix, pnorm, pcompact, args, return_qvalue=False, info=False)
            if len(target_peak) < 1: continue
            pred_peak = find_peaks(ppred_matrix, pnorm, pcompact, args, return_qvalue=False, info=False)
            
            pred_result, _ = peak_similarity(pred_peak, target_peak, info=False)

            pred_results.append(pred_result)

        results = {
            'f1_scores' : pred_results,
        }

        result_file = os.path.join(save_dir, 'results.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)


