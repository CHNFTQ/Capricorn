# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# our implementation of image-based metrics. including MSE, SSIM and PSNR.
# --------------------------------------------------------

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from math import log10, exp
from tqdm import tqdm
from HiC_evaluation.utils import *
from dataset_informations import *
from data_processing.Read_npz import read_npz
import logging
import os
import json

def gaussian(width, sigma):
    gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=3):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def get_ssim(pred_matrix, target_matrix, device="cpu", mask=None, patch_size=200, window_size=11):
    pred_matrix_torch = torch.from_numpy(pred_matrix).unsqueeze(0).unsqueeze(0)
    target_matrix_torch = torch.from_numpy(target_matrix).unsqueeze(0).unsqueeze(0)
    if mask is not None:
      mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    window = create_window(window_size, pred_matrix_torch.size()[1]).type_as(pred_matrix_torch).to(device)

    channel = pred_matrix_torch.size()[1]
    total_size = pred_matrix.shape[-1]
    patch_size = patch_size
    padding = window_size // 2

    ssim_all = 0
    for i in tqdm(range(0, total_size, patch_size)):
      for j in range(0, total_size, patch_size):
        pred_patch = pred_matrix_torch[:, :, max(i-padding, 0):min(i+patch_size+padding, total_size), max(j-padding, 0):min(j+patch_size+padding, total_size)].to(device)
        target_patch = target_matrix_torch[:, :, max(i-padding, 0):min(i+patch_size+padding, total_size), max(j-padding, 0):min(j+patch_size+padding, total_size)].to(device)
        if mask is not None:
          mask_patch = mask_torch[:, :, max(i-padding, 0):min(i+patch_size+padding, total_size), max(j-padding, 0):min(j+patch_size+padding, total_size)].to(device)
        
        mu1 = F.conv2d(pred_patch, window, padding=5, groups=channel)
        mu2 = F.conv2d(target_patch, window, padding=5, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred_patch * pred_patch, window, padding=5, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target_patch * target_patch, window, padding=5, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred_patch * target_patch, window, padding=5, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if mask is not None:
          ssim_map = (ssim_map * mask_patch).to("cpu")
        else:
          ssim_map = ssim_map.to("cpu")
        start_idx_i = min(i, padding)
        start_idx_j = min(j, padding)
        end_idx_i = min(start_idx_i + patch_size, patch_size)
        end_idx_j = min(start_idx_j + patch_size, patch_size)
        ssim_all += ssim_map[:, :, start_idx_i:end_idx_i, start_idx_j:end_idx_j].sum()
    if mask is not None:
      chr_ssim = ssim_all / mask.sum()
    else:
      chr_ssim = ssim_all / (total_size * total_size)
    return chr_ssim.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict-dir', type=str)
    parser.add_argument('--predict-resolution', type=str, default='10kb')
    parser.add_argument('--predict-caption', type=str, default='hic')

    parser.add_argument('-t', '--target-dir', type=str, default='/data/hic_data/multichannel_mat/HiC/GM12878')
    parser.add_argument('--target-resolution', type=str, default='10kb')
    parser.add_argument('--target-caption', type=str, default='hic')

    parser.add_argument('-c', '--cell-line', type=str, default='GM12878')
    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )
    
    parser.add_argument('--bound', type=int, default=200, help='Only evaluate the area within the bounding distance to diagonal')
    args = parser.parse_args()

    bound = args.bound

    mse_list = []
    ssim_list = []
    psnr_list = [] 

    save_dir = os.path.join(args.predict_dir, args.predict_resolution, 'Image_Metrics')
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

    chr_list = set_dict[args.dataset]
    abandon_chromosome = abandon_chromosome_dict.get(args.cell_line, [])

    for n in chr_list:
        if n in abandon_chromosome:
            continue

        pred_file = os.path.join(args.predict_dir, f'chr{n}_{args.predict_resolution}.npz')
        
        pred_matrix, _, _ = read_npz(pred_file, hic_caption = args.predict_caption, bound = bound, include_additional_channels=False)
        
        target_file = os.path.join(args.target_dir, f'chr{n}_{args.target_resolution}.npz')
        
        target_matrix, _, _ = read_npz(target_file, hic_caption = args.target_caption, bound = bound, include_additional_channels=False)

        if args.bound:
            mask = np.zeros_like(pred_matrix)
            
            for i in range(mask.shape[0]):
                for j in range(max(0, i-bound+1), min(mask.shape[1], i+bound)):
                    mask[i][j] = 1
            
            chr_mse = (((pred_matrix - target_matrix) ** 2) * mask).sum() / (mask.sum())
            chr_ssim = get_ssim(pred_matrix, target_matrix, mask=mask, device="cuda")
        else:
            chr_mse = ((pred_matrix - target_matrix) ** 2).mean()
            chr_ssim = get_ssim(pred_matrix, target_matrix, device=f"cuda")

        chr_psnr = 10 * log10(1 / (chr_mse + 1e-10))

        print_info("MSE for chr"+str(n)+":"+str(chr_mse))
        print_info("SSIM for chr"+str(n)+":"+str(chr_ssim))
        print_info("PSNR for chr"+str(n)+":"+str(chr_psnr))
        
        mse_list.append(chr_mse)
        ssim_list.append(chr_ssim)
        psnr_list.append(chr_psnr)

    print_info('MSE:')
    print_info(str(mse_list))
    print_info('SSIM:')
    print_info(str(ssim_list))
    print_info('PSNR:')
    print_info(str(psnr_list))
    print_info("Average MSE:")
    print_info(str(np.mean(mse_list)))
    print_info("Average SSIM:")
    print_info(str(np.mean(ssim_list)))
    print_info("Average PSNR:")
    print_info(str(np.mean(psnr_list)))
