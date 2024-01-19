import pandas as pd
import argparse
import numpy as np
from dataset_informations import *
from scipy import stats

def seg_similarity(pred_segs, tgt_segs, relation, info=True):
    # compute the f1 score of two peak sets
    if len(tgt_segs) <= 0: return -1

    predict_match = np.zeros(len(pred_segs))
    target_match = np.zeros(len(tgt_segs))
    
    for i, pred_seg in enumerate(pred_segs):
        for j, tgt_seg in enumerate(tgt_segs):   

            if relation == 'overlap':
                if pred_seg[0]>tgt_seg[1] or tgt_seg[0]>pred_seg[1]:
                    continue
                else:
                    predict_match[i] = 1
                    target_match[j] = 1
            elif relation == 'PinT':
                if pred_seg[0]>=tgt_seg[0] and pred_seg[1]<=tgt_seg[1]:
                    predict_match[i] = 1
                    target_match[j] = 1
                else:
                    continue
            elif relation == 'TinP':
                if tgt_seg[0]>=pred_seg[0]  and tgt_seg[1]<=pred_seg[1] :
                    predict_match[i] = 1
                    target_match[j] = 1
                else:
                    continue
        
    pred_matched = np.sum(predict_match)
    tgt_matched = np.sum(target_match)

    precision = np.mean(predict_match)
    recall = np.mean(target_match)

    f1 = 2  / (1/ precision + 1/recall)

    if info:
        print(f'pred matched: {pred_matched}, tgt matched: {tgt_matched}, predicted: {len(pred_segs)}, target: {len(tgt_segs)}')

    return pred_matched, tgt_matched, precision, recall, f1

def read_bed_file(file_name, extend = 0):
    data = pd.read_csv(file_name,sep='\t')

    segs = {}
    for index, row in data.iterrows():
        
        chrom = row['chrom']
        if 'chr' not in chrom:
            chrom = 'chr'+chrom

        if chrom not in segs:
            segs[chrom] = []

        segs[chrom].append([row['start']-extend, row['end']+extend])
    
    return segs

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict-bed-file', type=str)
    parser.add_argument('-t', '--target-bed-file', type=str)

    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('-e', '--extend-segment', type=float, default = 0)

    parser.add_argument('-r', '--relation', type=str, choices=['overlap', 'TinP', 'PinT'])

    parser.add_argument('-o', '--output-file', type=str, default = None)
    args = parser.parse_args()

    extend_segment = args.extend_segment
    relation = args.relation
    output_file = args.output_file

    predict_segs = read_bed_file(args.predict_bed_file, extend = extend_segment)
    target_segs = read_bed_file(args.target_bed_file, extend = extend_segment)
    
    chr_list = set_dict[args.dataset]

    results = []
    for n in chr_list:

        chr = f'chr{n}'
        if chr not in target_segs:
            print(f'{chr}: no target. Skip.' )
            continue

        if chr not in predict_segs:
            results.append((chr, 0, len(target_segs[chr]), 0, 0, 0, 0, 0))
            continue

        pred_matched, target_matched, precision, recall, f1 = seg_similarity(
            predict_segs[chr], 
            target_segs[chr], 
            relation = relation, 
            )
        
        results.append((chr, len(predict_segs[chr]), len(target_segs[chr]), pred_matched, target_matched, precision, recall, f1))
        print(f'{chr} matched: precision: {precision}, recall: {recall}, f1: {f1}')
    
    chrs, num_pred, num_target, pred_matched, target_matched, precisions, recalls, f1s = list(map(list, zip(*results)))
    
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

        chrs.append('sum')
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

