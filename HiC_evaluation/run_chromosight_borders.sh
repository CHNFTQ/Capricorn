#!/bin/bash

chromosight detect --pattern=borders \
                   --pearson=0.4 \
                   $1/$2 \
                   $1/$3_borders

chromosight detect --pattern=hairpins \
                   --pearson=0.4 \
                   $1/$2 \
                   $1/$3_hairpins