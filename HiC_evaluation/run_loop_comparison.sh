#!/bin/bash

TARGET_DIR=/data/hic_data/mat/K562/10kb
PRED_DIR=/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/K562_HiCSR_GM12878_fixed/10kb

SAVE_DIR=results/HiCSR/K562

mkdir -p $SAVE_DIR

python -m HiC_evaluation.bedpe_comparison -t $TARGET_DIR/HiCCUPS/HiCCUPS_loop_annotation.bedpe -p $PRED_DIR/HiCCUPS/HiCCUPS_loop_annotation.bedpe -d -o $SAVE_DIR/HiCCUPS.tsv

python -m HiC_evaluation.bedpe_comparison -t $TARGET_DIR/chromosight_small.tsv -p $PRED_DIR/chromosight_small.tsv -d -o $SAVE_DIR/chromosight.tsv

python -m HiC_evaluation.bedpe_comparison -t $TARGET_DIR/mustache.tsv -p $PRED_DIR/mustache.tsv -d -o $SAVE_DIR/mustache.tsv