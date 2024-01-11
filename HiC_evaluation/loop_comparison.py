import pandas as pd
import argparse
import numpy as np
from HiC_evaluation.utils import *
from dataset_informations import *

def peak_similarity(pred_peaks, tgt_peaks, matching_scope=5.0, norm_ord = 2, dynamic_scope = False ,info=True):
    # compute the f1 score of two peak sets
    if len(tgt_peaks) <= 0: return -1

    predict_match = np.zeros(len(pred_peaks))
    target_match = np.zeros(len(tgt_peaks))
    
    for i, pred_peak in enumerate(pred_peaks):
        for j, tgt_peak in enumerate(tgt_peaks):
            pred_idx = np.array(pred_peak)
            tgt_idx = np.array(tgt_peak)
            
            if dynamic_scope:
                # dynamic scope at near-diagonal area
                # More strict about near-diagonal loops
                scope = min(matching_scope, 0.2 * abs(pred_idx[0]-pred_idx[1]))
            else:
                scope = matching_scope

            if np.linalg.norm(pred_idx - tgt_idx, norm_ord) <= scope:
                predict_match[i] = 1
                target_match[j] = 1
    
    pred_matched = np.sum(predict_match)
    tgt_matched = np.sum(target_match)

    precision = np.mean(predict_match)
    recall = np.mean(target_match)

    f1 = 2  / (1/ precision + 1/recall)

    if info:
        print_info(f'pred matched: {pred_matched}, tgt matched: {tgt_matched}, predicted: {len(pred_peaks)}, target: {len(tgt_peaks)}')

    return pred_matched, tgt_matched, precision, recall, f1

def read_loop_file(file_name):
    data = pd.read_csv(file_name,sep='\t')

    loops = {}
    for index, row in data.iterrows():
        
        if row['chrom1'] not in loops:
            loops[row['chrom1']] = []
        
        center = ((row['start1']+row['end1'])/2, (row['start2']+row['end2'])/2)

        # assert abs(center[1] - center[0]) <= 2000000

        loops[row['chrom1']].append(center)
    
    loops.pop('chrM', None)

    return loops

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict-loop-file', type=str)
    parser.add_argument('-t', '--target-loop-file', type=str)

    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('-m', '--matching-scope', type=float, default = 50000)
    parser.add_argument('-n', '--norm-ord', type=float, default = 2)
    parser.add_argument('-d', '--dynamic-scope', action='store_true')
    args = parser.parse_args()

    matching_scope = args.matching_scope
    norm_ord = args.norm_ord
    dynamic_scope = args.dynamic_scope

    if norm_ord >= 10:
        norm_ord = np.inf
    elif norm_ord <= -10:
        norm_ord = -np.inf

    predict_loops = read_loop_file(args.predict_loop_file)
    target_loops = read_loop_file(args.target_loop_file)
    
    chr_list = set_dict[args.dataset]

    results = []
    for n in chr_list:
        chr = f'chr{n}'
        if chr not in predict_loops:
            print(f'{chr}: no predict' )
            continue
        if chr not in target_loops:
            print(f'{chr}: no target' )
            continue

        pred_matched, target_matched, precision, recall, f1 = peak_similarity(
            predict_loops[chr], 
            target_loops[chr], 
            matching_scope = matching_scope, 
            norm_ord = norm_ord,
            dynamic_scope = dynamic_scope
            )
        
        results.append((len(predict_loops[chr]), len(target_loops[chr]), pred_matched, target_matched, precision, recall, f1))
        print(f'{chr} matched: precision: {precision}, recall: {recall}, f1: {f1}')
    
    num_pred, num_target, pred_matched, target_matched, precisions, recalls, f1s = zip(*results)
    
    sum_pred = sum(num_pred)
    sum_target = sum(num_target)
    sum_pred_matched = sum(pred_matched)
    sum_target_matched = sum(target_matched)
    
    print(f'sum predict: {sum_pred}, sum target: {sum_target}')
    print(f'sum predict matched: {sum_pred_matched}, sum target matched: {sum_target_matched}')

    precision = sum_pred_matched/sum_pred
    recall = sum_target_matched/sum_target

    print(f'overall precision: {precision}, recall: {recall}, f1: {2/(1/precision+1/recall)}')
    print(f'average precision: {np.mean(precisions)}, recall: {np.mean(recalls)}, f1: {np.mean(f1s)}')
    
