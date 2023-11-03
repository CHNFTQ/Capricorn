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
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-titles', nargs='+', type=str, default=['Diffusion(3d)', 'Diffusion(2d)'])
    parser.add_argument('--dataset', type=str, default='GM12878')

    args = parser.parse_args()
    chr_list = dataset_chrs[args.dataset]

    save_dir = os.path.join('Figures/Fig3/plots', args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    pred_file = os.path.join('Figures/Fig3', args.method_titles[-2], args.dataset, 'results.json')
    baseline_file = os.path.join('Figures/Fig3', args.method_titles[-1], args.dataset, 'results.json')

    with open(pred_file, 'r') as f:
        pred_results=json.load(f)['f1_scores']
    with open(baseline_file, 'r') as f:
        baseline_results=json.load(f)['f1_scores']

    x = []
    y = []

    num_pred_better = 0
    num_baseline_better = 0

    for pr, br in zip(pred_results, baseline_results):
        pr = max(pr, 0)
        br = max(br, 0)
        x.append(pr)
        y.append(br)
        if pr > br:
            num_pred_better += 1
        if pr < br:
            num_baseline_better += 1

    print_info(f'predict mean: {np.mean(pred_results)}, baseline mean: {np.mean(baseline_results)}')
    print_info(f'points number: {len(x)}, pred better number: {num_pred_better}, baseline better number: {num_baseline_better}')

    max_f1 = 1
    min_f1 = 0
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.scatter(x,y, s=5)

    ax.set_xlabel(args.method_titles[-2])
    ax.set_xlim(min_f1, max_f1)
    ax.set_ylabel(args.method_titles[-1])
    ax.set_ylim(min_f1, max_f1)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    file_name = os.path.join(save_dir, f"f1_score_{args.dataset}_{'_'.join(args.method_titles)}_0.pdf")
    plt.savefig(file_name)
    plt.clf()

    fig, ax = plt.subplots()
    using_hist2d(ax, x, y, ((min_f1, max_f1), (min_f1, max_f1)), bins=(50, 50))

    ax.set_xlabel(args.method_titles[-2])
    # ax.set_xlim(min_f1, max_f1)
    ax.set_ylabel(args.method_titles[-1])
    # ax.set_ylim(min_f1, max_f1)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    file_name = os.path.join(save_dir, f"f1_score_{args.dataset}_{'_'.join(args.method_titles)}_1.pdf")
    plt.savefig(file_name)
    plt.clf()

    fig, ax = plt.subplots()
    ax = density_scatter(x, y)

    ax.set_xlabel(args.method_titles[-2])
    ax.set_xlim(min_f1, max_f1)
    ax.set_ylabel(args.method_titles[-1])
    ax.set_ylim(min_f1, max_f1)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    file_name = os.path.join(save_dir, f"f1_score_{args.dataset}_{'_'.join(args.method_titles)}_2.pdf")
    plt.savefig(file_name)
    plt.clf()

    incs = []
    for pr, br in zip(pred_results, baseline_results):
        pr = max(pr, 0)
        br = max(br, 0)
        incs.append(pr-br)
    
    fig, ax = plt.subplots()
    
    counts, bins = np.histogram(incs, range=(-max_f1, max_f1), bins=21)
    ax.hist(bins[:-1], bins, weights=counts)

    file_name = os.path.join(save_dir, f"f1_score_{args.dataset}_{'_'.join(args.method_titles)}_hist.pdf")
    plt.savefig(file_name)
    plt.clf()





