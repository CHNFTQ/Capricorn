#!/bin/bash

HiC_CAPTION=hicarn

if ! test -f "$1/bound200_10kb.cool"; then
    echo "creating cool file"

    rename predict_c c $1/*.npz
    rename 40kb 10kb $1/*.npz

    python -m data_processing.npz2cool --data-dir $1 --hic-caption $HiC_CAPTION
fi

python -m mustache -r 10kb -pt 0.05 -f $1/bound200_10kb.cool -o $1/mustache.tsv