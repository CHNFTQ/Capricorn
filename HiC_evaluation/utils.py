# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# utilities for HiC matrix evaluation.
# --------------------------------------------------------

import numpy as np
import os
import logging

def get_oe_matrix(matrix, bounding = 100000000, oe=True):
    max_offset = min(matrix.shape[0], bounding)
    
    expected = [np.mean(np.diagonal(matrix, offset)) for offset in range(max_offset)]
    
    if oe:
        e_matrix = np.zeros_like(matrix, dtype=np.float32)
        oe_matrix = np.zeros_like(matrix, dtype=np.float32)
        for i in range(matrix.shape[0]):
            for j in range(max(i-bounding+1, 0), min(i+bounding, matrix.shape[1])):
                e_matrix[i][j] = expected[abs(i-j)]
                oe_matrix[i][j] = matrix[i][j]/expected[abs(i-j)] if expected[abs(i-j)] != 0 else 0

        return oe_matrix, e_matrix
    else:
        return expected

def read_matrices(args, chr, no_bounding = False, compact_idx = False):
    pred_data = np.load(os.path.join(args.predict_dir, f"{args.predict_prefix}chr{chr}_{args.predict_resolution}.npz"))
    pred_matrix = pred_data[args.predict_caption]
    target_data = np.load(os.path.join(args.target_dir, f"{args.target_prefix}chr{chr}_{args.target_resolution}.npz"))
    target_matrix = target_data[args.target_caption]
    
    if not no_bounding and args.bounding is not None:
        mask = np.zeros((target_matrix.shape[-2], target_matrix.shape[-1]))
        for i in range(mask.shape[0]):
            for j in range(max(i-args.bounding+1, 0), min(i+args.bounding, mask.shape[1])):         
                mask[i][j] = 1
        
        pred_matrix = pred_matrix * mask
        target_matrix = target_matrix * mask
    if compact_idx:
        return pred_matrix, target_matrix, pred_data['compact']
    else:
        return pred_matrix, target_matrix

def print_info(s:str):
    print(s)
    logging.info(s)
