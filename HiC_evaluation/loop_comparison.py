import pandas as pd
import argparse
import numpy as np
from HiC_evaluation.utils import *

def count_num(peaks):
    num = 0
    for k,v in peaks.items():
        num += len(v)

    return num

def peak_similarity(pred_peaks, tgt_peaks, matching_scope=5.0, info=True):
    # compute the f1 score of two peak sets
    predicted = len(pred_peaks)
    target = len(tgt_peaks)
    if target <= 0: return -1
    
    matched = 0
    for pred_peak in pred_peaks:
        for tgt_peak in tgt_peaks:
            pred_idx = np.array(pred_peak)
            tgt_idx = np.array(tgt_peak)
            scope = min(matching_scope, 0.2 * abs(pred_idx[0]-pred_idx[1]))
            if np.linalg.norm(pred_idx - tgt_idx) <= scope:
                matched += 1
                break

    if info:
        print_info(f'matched: {matched}, predicted: {predicted}, target: {target}')
    
    precision = matched / predicted if predicted else 0
    recall = matched / target if target else 0

    f1 = 2 * matched / (predicted + target) #equivalent to 2/(1/precision+1/recall)
    return matched, precision, recall, f1

def read_loop_file(file_name):
    data = pd.read_csv(file_name,sep='\t')

    loops = {}
    for index, row in data.iterrows():
        
        if row['chrom1'] not in loops:
            loops[row['chrom1']] = []
        
        center = ((row['start1']+row['end1'])/2, (row['start2']+row['end2'])/2)

        assert abs(center[1] - center[0]) <= 2000000

        loops[row['chrom1']].append(center)
    
    return loops

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict-loop-file', type=str)
    parser.add_argument('-t', '--target-loop-file', type=str)

    parser.add_argument('--matching-scope', type=float, default = 50000)
    args = parser.parse_args()

    predict_loops = read_loop_file(args.predict_loop_file)
    target_loops = read_loop_file(args.target_loop_file)

    sum_pred = count_num(predict_loops)
    sum_tgt = count_num(target_loops)
    print(f'number of predict loops: {sum_pred}')
    print(f'number of target loops: {sum_tgt}')

    chrs = set(list(predict_loops.keys()) + list(target_loops.keys()))
    results = []
    for n in chrs:
        matched, precision, recall, f1 = peak_similarity(predict_loops[n], target_loops[n], matching_scope = args.matching_scope)
        results.append((matched, precision, recall, f1))
        print(f'matched: {matched}, precision: {precision}, recall: {recall}, f1: {f1}')
    
    matched, precision, recall, f1 = zip(*results)
    sum_matched = sum(matched)
    print(f'sum matched: {sum_matched}')
    print(f'overall precision: {sum_matched/sum_pred}, recall: {sum_matched/sum_tgt}, f1: {2*sum_matched/(sum_pred+sum_tgt)}')
    print(f'average precision: {np.mean(precision)}, recall: {np.mean(recall)}, f1: {np.mean(f1)}')
    
