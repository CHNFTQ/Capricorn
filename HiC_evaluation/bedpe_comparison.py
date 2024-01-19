import pandas as pd
import argparse
import numpy as np
from HiC_evaluation.utils import *
from dataset_informations import *
from scipy import stats

def area_similarity(pred_peaks, tgt_peaks, matching_scope=5.0, norm_ord = 2, dynamic_scope = False ,info=True):
    # compute the f1 score of two peak sets
    if len(tgt_peaks) <= 0: return -1

    predict_match = np.zeros(len(pred_peaks))
    target_match = np.zeros(len(tgt_peaks))
    
    for i, pred_peak in enumerate(pred_peaks):
        for j, tgt_peak in enumerate(tgt_peaks):
            pred_idx = np.array(pred_peak)
            tgt_idx = np.array(tgt_peak)
            
            if dynamic_scope:
                # dynamic scope when near-diagonal
                # More strict about near-diagonal areas
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

def read_bedpe_file(file_name):
    data = pd.read_csv(file_name,sep='\t')

    areas = {}
    for index, row in data.iterrows():
        chrom = row['chrom1']
        if 'chr' not in chrom:
            chrom = 'chr'+chrom

        if chrom not in areas:
            areas[chrom] = []
        
        center = ((row['start1']+row['end1'])/2, (row['start2']+row['end2'])/2)

        # assert abs(center[1] - center[0]) <= 2000000

        areas[chrom].append(center)
    
    areas.pop('chrM', None)

    return areas

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict-bedpe-file', type=str)
    parser.add_argument('-t', '--target-bedpe-file', type=str)

    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('-m', '--matching-scope', type=float, default = 50000)
    parser.add_argument('-n', '--norm-ord', type=float, default = 2)
    parser.add_argument('-d', '--dynamic-scope', action='store_true')

    parser.add_argument('-o', '--output-file', type=str, default = None)
    args = parser.parse_args()

    matching_scope = args.matching_scope
    norm_ord = args.norm_ord
    dynamic_scope = args.dynamic_scope
    output_file = args.output_file

    if norm_ord >= 10:
        norm_ord = np.inf
    elif norm_ord <= -10:
        norm_ord = -np.inf

    predict_areas = read_bedpe_file(args.predict_bedpe_file)
    target_areas = read_bedpe_file(args.target_bedpe_file)
    
    chr_list = set_dict[args.dataset]

    results = []
    for n in chr_list:

        chr = f'chr{n}'
        if chr not in target_areas:
            print(f'{chr}: no target. Skip.' )
            continue

        if chr not in predict_areas:
            results.append((chr, 0, len(target_areas[chr]), 0, 0, 0, 0, 0))
            continue

        pred_matched, target_matched, precision, recall, f1 = area_similarity(
            predict_areas[chr], 
            target_areas[chr], 
            matching_scope = matching_scope, 
            norm_ord = norm_ord,
            dynamic_scope = dynamic_scope
            )
        
        results.append((chr, len(predict_areas[chr]), len(target_areas[chr]), pred_matched, target_matched, precision, recall, f1))
        print(f'{chr} matched: precision: {precision}, recall: {recall}, f1: {f1}')
    
    chrs, num_pred, num_target, pred_matched, target_matched, precisions, recalls, f1s = list(map(list, zip(*results)))
    
    sum_pred = sum(num_pred)
    sum_target = sum(num_target)
    sum_pred_matched = int(sum(pred_matched))
    sum_target_matched = int(sum(target_matched))
    
    print(f'sum predict: {sum_pred}, sum target: {sum_target}')
    print(f'sum predict matched: {sum_pred_matched}, sum target matched: {sum_target_matched}')

    precision = sum_pred_matched/sum_pred
    recall = sum_target_matched/sum_target
    f1 = 2/(1/precision+1/recall)

    print(f'overall precision: {precision}, recall: {recall}, f1: {f1}')
    print(f'average precision: {np.mean(precisions)}, recall: {np.mean(recalls)}, f1: {np.mean(f1s)}')

    if output_file is not None:

        chrs.append('average')
        num_pred.append(np.mean(num_pred))
        num_target.append(np.mean(num_target))
        pred_matched.append(np.mean(pred_matched))
        target_matched.append(np.mean(target_matched))
        precisions.append(np.mean(precisions))
        recalls.append(np.mean(recalls))
        f1s.append(np.mean(f1s))

        chrs.append('standard error')
        num_pred.append(stats.sem(num_pred[:-1]))
        num_target.append(stats.sem(num_target[:-1]))
        pred_matched.append(stats.sem(pred_matched[:-1]))
        target_matched.append(stats.sem(target_matched[:-1]))
        precisions.append(stats.sem(precisions[:-1]))
        recalls.append(stats.sem(recalls[:-1]))
        f1s.append(stats.sem(f1s[:-1]))

        chrs.append('overall')
        num_pred.append(sum_pred)
        num_target.append(sum_target)
        pred_matched.append(sum_pred_matched)
        target_matched.append(sum_target_matched)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        result_data = pd.DataFrame.from_dict({
            'chromosome' : chrs,
            'predicted' : num_pred,
            'target' : num_target,
            'matched predicted' : pred_matched,
            'matched target' : target_matched,
            'precision' : precisions,
            'recall' : recalls,
            'F1 score': f1s
            })
        result_data.to_csv(output_file, sep='\t', index = False)

