#!/bin/bash

HiC_CAPTION=hicarn

if ! test -f "$1/bound200_10kb.cool"; then
    echo "creating cool file"

    rename predict_c c $1/*.npz
    rename 40kb 10kb $1/*.npz

    python -m data_processing.npz2cool --data-dir $1 --hic-caption $HiC_CAPTION
fi

chromosight detect --min-dist 20000 --max-dist 200000 $1/bound200_10kb.cool $1/chromosight