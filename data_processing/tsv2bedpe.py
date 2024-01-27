# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to transform tsv outputs for loops to bedpe format
# --------------------------------------------------------

import pandas as pd
import argparse

required_domain = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'name', 'score', 'strand1', 'strand2', 'color']

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--tsv-file', type=str)
    parser.add_argument('-o', '--output-file', type=str, default = None)
    parser.add_argument('-c', '--default-color', type=str, default = '0,0,255')
    args = parser.parse_args()

    tsv_file = args.tsv_file
    output_file = args.output_file

    default_color = args.default_color

    if output_file is None:
        output_file = args.tsv_file.split('.')[0]+'.bedpe'

    data = pd.read_csv(tsv_file,sep='\t')

    n = len(data)

    default = ['.'] * n

    bedpe_data = {}

    for domain in required_domain:
        if domain in data.columns:
            bedpe_data[domain] = data[domain]
        else:
            if domain == 'color':
                bedpe_data[domain] = default_color
            else:
                bedpe_data[domain] = default
    
    for domain in data.columns:
        if domain not in required_domain:
            bedpe_data[domain] = data[domain]
    
    pd.DataFrame(bedpe_data).to_csv(output_file, sep='\t', index=False)

            
    

