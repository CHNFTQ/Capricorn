# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# our implementation of the insulation score algorithm.
# --------------------------------------------------------

import argparse
import numpy as np
import os
from HiC_evaluation.utils import *
from HiC_evaluation.dataset_info import *
from HiC_evaluation.args import *
import numpy as np
import logging
from utils import compactM
import sys
eps=1e-20

def compute_insulation_score(matrix, window_size, extra_channel = None):
    if len(matrix.shape) == 2:
        L, _ = matrix.shape
        scores = []
        for i in range(L):
            if i<window_size or i+window_size >= L: scores.append(None)
            else:
                scores.append(np.mean(matrix[i-window_size:i, i+1:i+window_size+1]))

    elif len(matrix.shape) == 3 and extra_channel is not None:
        # use the extra channel as score.
        C, L, _ = matrix.shape
        scores = []
        for i in range(L):
            if i<window_size or i+window_size >= L: scores.append(None)
            else:
                if matrix[extra_channel, i-window_size, i+window_size] > 0:
                    scores.append(matrix[1, i-window_size, i+window_size])
                else:
                    # extra-channel not available, use original instead
                    scores.append(np.mean(matrix[0, i-window_size:i, i+1:i+window_size+1]))
            
    return scores

def compute_bounds(insulation_scores, delta_smooth_size, bound_strength):
    L = len(insulation_scores)
    
    mean_score = np.mean([s for s in insulation_scores if s is not None])
        
    normalized_scores = []
    for score in insulation_scores:
        if score is not None:
            normalized_scores.append((np.log(score+eps) - np.log(mean_score+eps))/np.log(2))
        else:
            normalized_scores.append(None)
    
    delta = []
    for i in range(L):
        left = []
        right = []
        for j in range(delta_smooth_size):
            if i+1+j<L and normalized_scores[i+1+j] is not None:
                right.append(normalized_scores[i+1+j])
            if i-j>=0 and normalized_scores[i-j] is not None:
                left.append(normalized_scores[i-j])
        if len(left) == 0 or len(right) == 0:
            delta.append(None)
        else:
            delta.append(np.mean(right) - np.mean(left))
        
    minimas = []
    for i in range(L):
        if i==0 or delta[i-1] is None or delta[i] is None: continue
        if delta[i-1] < 0 and delta[i] >0:
            minimas.append(i)

    # print(np.around(insulation_scores[5420:5470], decimals=4))
    # print(np.around(normalized_scores[5420:5470], decimals=4))
    # print(np.around(delta[5420:5470], decimals=4))

    bounds = []
    for minima in minimas:
        l = minima-1
        while delta[l-1] is not None and delta[l-1] <= delta[l]:
            l -= 1
        
        r = minima
        while delta[r+1] is not None and delta[r+1] >= delta[r]:
            r += 1

        if delta[r] - delta[l] >= bound_strength:
            bounds.append(minima)
        
    return bounds

def compute_TAD_similarity(pred_matrix, tgt_matrix, args):
    tgt_iscores = compute_insulation_score(tgt_matrix, args.window_size)
    tgt_bounds = compute_bounds(tgt_iscores, args.delta_smooth_size, args.bound_strength)

    pred_iscores = compute_insulation_score(pred_matrix, args.window_size, args.extra_channels[0])
    pred_bounds = compute_bounds(pred_iscores, args.delta_smooth_size, args.bound_strength)

    diff = []
    for ps, ts in zip(pred_iscores, tgt_iscores):
        if ts is not None:
            assert ps is not None
            diff.append(ps-ts)
    insu_mse = np.mean(np.square(diff))
    insu_diff_norm = np.linalg.norm(diff, ord=2)

    matched = 0

    ti = iter(tgt_bounds)
    tb = next(ti, None)
    for pb in pred_bounds:
        while tb is not None and tb < pb-args.boundary_zone_size:
            # np.set_printoptions(threshold=sys.maxsize)
            # print(f'T:{tb} (P:{pb}) unmatch')
            # s = 8
            # print(np.around(tgt_iscores[tb-s:tb+s+1], decimals=4))
            # print(np.array2string(tgt_matrix[tb-s:tb+s+1, tb-s:tb+s+1], precision=1, floatmode='fixed'))
            # print(np.around(pred_iscores[tb-s:tb+s+1], decimals=4))
            # print(pred_matrix[tb-s:tb+s+1, tb-s:tb+s+1])
            
            tb = next(ti, None)

        if tb is None: break

        if abs(pb-tb)<=args.boundary_zone_size:
            matched += 1
            tb = next(ti, None)
        # else:
        #     np.set_printoptions(threshold=sys.maxsize)
        #     print(f'P:{pb} (T:{tb}) unmatch')
            # s = 8
            # print(np.around(tgt_iscores[pb-s:pb+s+1], decimals=4))
            # print(np.array2string(tgt_matrix[pb-s:pb+s+1, pb-s:pb+s+1], precision=1, floatmode='fixed'))
            # print(np.around(pred_iscores[pb-s:pb+s+1], decimals=4))
            # print(pred_matrix[pb-s:pb+s+1, pb-s:pb+s+1])

    
    lp = len(pred_bounds)
    lt = len(tgt_bounds)

    print_info(f'predicted bounds: {lp}, target bounds: {lt}, matched bounds: {matched}')
    return 2*matched/(lp+lt), insu_mse, insu_diff_norm

if __name__ == '__main__':
    parser = evaluate_parser()
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--delta-smooth-size', type=int, default=10)
    parser.add_argument('--bound-strength', type=float, default=0.1)
    parser.add_argument('--boundary-zone-size', type=int, default=3)
    
    parser.add_argument('--use-extra-channels', action='store_true')
    parser.add_argument('--extra-channels', nargs='+', type=int, default=[2])

    args = parser.parse_args()
    f1_scores = []
    insu_mses = []
    insu_diff_norms = []

    extrachannel_f1_scores = []
    extrachannel_insu_mses = []
    extrachannel_insu_diff_norms = []

    save_dir = os.path.join(args.predict_dir, 'TAD' if args.save_name is None else args.save_name)
    os.makedirs(save_dir, exist_ok=True)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    for chr in dataset_chrs[args.dataset]:
        full_pred_matrix, full_target_matrix, compact_idx = read_matrices(args, chr, compact_idx = True)

        full_pred_matrix = compactM(full_pred_matrix, compact_idx)
        full_target_matrix = compactM(full_target_matrix, compact_idx)

        if len(full_pred_matrix.shape)>=3 :
            pred_matrix = full_pred_matrix[0]
        else:
            pred_matrix = full_pred_matrix
        if len(full_target_matrix.shape)>=3 :
            target_matrix = full_target_matrix[0]
        else:
            target_matrix = full_target_matrix

        f1_score, insu_mse, insu_diff_norm = compute_TAD_similarity(pred_matrix, target_matrix, args)
        print_info(f"TAD f1 score for chr {chr} : {f1_score}")
        print_info(f"Insulation score MSE for chr {chr} : {insu_mse}")
        print_info(f"Insulation score difference norm for chr {chr} : {insu_diff_norm}")
        f1_scores.append(f1_score)
        insu_mses.append(insu_mse)
        insu_diff_norms.append(insu_diff_norm)

        if args.use_extra_channels:
            f1_score, insu_mse, insu_diff_norm = compute_TAD_similarity(full_pred_matrix[ args.extra_channels[0] ], target_matrix, args)
            print_info(f"[with extra channels]TAD f1 score for chr {chr} : {f1_score}")
            print_info(f"[with extra channels]Insulation score MSE for chr {chr} : {insu_mse}")
            print_info(f"[with extra channels]Insulation score difference norm for chr {chr} : {insu_diff_norm}")
            extrachannel_f1_scores.append(f1_score)
            extrachannel_insu_mses.append(insu_mse)
            extrachannel_insu_diff_norms.append(insu_diff_norm)

    print_info(f"TAD f1 scores:{f1_scores}")
    print_info(f"insulation score MSE:{insu_mses}")
    print_info(f"insulation score difference norm:{insu_diff_norms}")    

    print_info(f"Average TAD f1 score:{np.mean(f1_scores):.4f}")
    print_info(f"Average insulation score MSE:{np.mean(insu_mses):.4e}")
    print_info(f"Average insulation score difference norm:{np.mean(insu_diff_norms):.4e}")

    if args.use_extra_channels:
        print_info(f"[with extra channels]TAD f1 score1s:{extrachannel_f1_scores}")
        print_info(f"[with extra channels]insulation score MSE:{extrachannel_insu_mses}")
        print_info(f"[with extra channels]insulation score difference norm:{insu_diff_norms}")    

        print_info(f"[with extra channels]Average TAD f1 score :{np.mean(extrachannel_f1_scores):.4f}")
        print_info(f"[with extra channels]Average insulation score MSE:{np.mean(extrachannel_insu_mses):.4e}")
        print_info(f"[with extra channels]Average insulation score difference norm:{np.mean(insu_diff_norms):.4e}")
