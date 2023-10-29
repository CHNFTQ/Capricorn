# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the HiCARN implementation, a script to generate data file for multichannel matrices. Also fix bugs to ensure locuses near edges are covered.
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------

import sys
from typing import Any
import numpy as np
from Arg_Parser import *
from utils import *
from tqdm import tqdm

except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}

def divide_multichannel(mat, chr_num, chunk_size=40, stride=40, bound=200, diagonal_stride = 40, padding=True, species='hsa', verbose=False):
    """
    Dividing method.
    """
    chr_str = str(chr_num)
    if isinstance(chr_num, str): chr_num = except_chr[species][chr_num]
    result = []
    index = []

    if len(mat.shape) <= 2:
        mat = np.expand_dims(mat, axis=0)

    channel, size, _ = mat.shape

    # if (diagonal_stride < chunk_size and padding):
    #     pad_len = (chunk_size - diagonal_stride) // 2
    #     mat = np.pad(mat, ((0,0), (pad_len, pad_len), (pad_len, pad_len)), 'constant')

    _, height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'

    assert diagonal_stride <= stride

    for i in range(0, height - chunk_size + diagonal_stride, diagonal_stride):
        i = i if i + chunk_size <= height else height - chunk_size
        for j in range(i, i + bound - chunk + 2*stride, stride):
            j = min(j, width - chunk_size)
            subImage = mat[..., i:i + chunk_size, j:j + chunk_size]
            result.append(subImage)
            index.append((chr_num, size, i, j))
            if j + chunk_size >= width : break
        for j in range(i - stride, i - bound + chunk - 2 * stride, - stride):
            j = max(j, 0)
            subImage = mat[..., i:i + chunk_size, j:j + chunk_size]
            result.append(subImage)
            index.append((chr_num, size, i, j))
            if j <= 0 : break

    result = np.array(result)
    if verbose: print(
        f'[Chr{chr_str}] Deviding HiC matrix ({channel}x{size}x{size}) into {len(result)} samples with chunk={chunk_size}, '
        f'stride={stride}, bound={bound} and diagonal_stride={diagonal_stride}')
    
    index = np.array(index)
    
    return result, index

def carn_divider(n, 
                 high_file, 
                 down_file,  
                 chunk=40, 
                 stride=40, 
                 bound=200,
                 diagonal_stride = 40,
                 pool_type='max', 
                 scale=1):
    hic_data = np.load(high_file)
    down_data = np.load(down_file)
    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[-1]

    hic = hic_data['hic']
    down_hic = down_data['hic']

    print(f'[Chr{n}]File loaded.')

    # Compacting
    hic = compactM(hic, compact_idx)
    down_hic = compactM(down_hic, compact_idx)
    print(f'[Chr{n}]Compacted.')
    # Deviding and Pooling    
    div_dhic, div_inds = divide_multichannel(down_hic, n, chunk, stride, bound, diagonal_stride)
    div_dhic = pooling(div_dhic, scale, pool_type=pool_type, verbose=False).numpy()

    div_hhic, _ = divide_multichannel(hic, n, chunk, stride, bound, diagonal_stride, verbose=True)
    print(f'[Chr{n}]Finished.')
    return n, div_dhic, div_hhic, div_inds, compact_idx, full_size

if __name__ == '__main__':
    parser = data_divider_parser()
    parser.add_argument('--diagonal-stride', type=int, default = 40, help='Allow the submatrices moving along the diagonal to obatin more data. Default: 40, equals to the stride, which means no additional move along diagonal')
    parser.add_argument('--transform-names', type=str, nargs='+', default = ['HiC', 'OE', '01TAD', 'Lp', 'Lr'], help='List of transforms used. Group transforms should be in the correct order(i.e. the next of Lp should be Lr)')
    parser.add_argument('--save-prefix', type=str, default = 'Multi')
    args = parser.parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    lr_cutoff = args.lr_cutoff
    dataset = args.dataset

    chunk = args.chunk
    stride = args.stride
    bound = args.bound
    diagonal_stride = args.diagonal_stride
    scale = args.scale
    pool_type = args.pool_type

    trs = args.transform_names

    prefix = args.save_prefix

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict[cell_line]
    postfix = cell_line.lower() if dataset == 'all' else dataset
    pool_str = 'nonpool' if scale == 1 else f'{pool_type}pool{scale}'
    print(f'Going to read {high_res} and {low_res} data with {trs}, then deviding matrices with {pool_str}')

    # pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    data_dir = os.path.join(root_dir, 'multichannel_mat', '_'.join(trs), cell_line)
    out_dir = os.path.join(root_dir, 'data')
    mkdir(out_dir)

    # start = time.time()
    # pool = multiprocessing.Pool(processes=pool_num)
    # print(f'Start a multiprocess pool with processes = {pool_num} for generating HiCARN data')
    results = []
    for n in tqdm(chr_list):
        if n in abandon_chromosome:
            continue
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
        kwargs = {'scale':scale, 'pool_type':pool_type, 'chunk':chunk, 'stride':stride, 'bound':bound, 'diagonal_stride' : diagonal_stride}
        res = carn_divider(n, high_file, down_file, **kwargs)
        results.append(res)
    # return: n, div_dhic, div_hhic, div_inds, compact_idx, full_size
    data = np.concatenate([r[1] for r in results])
    target = np.concatenate([r[2] for r in results])
    inds = np.concatenate([r[3] for r in results])
    compacts = {r[0]: r[4] for r in results}
    sizes = {r[0]: r[5] for r in results}

    filename = f'{prefix}_{high_res}{low_res}_c{chunk}_s{stride}_ds{diagonal_stride}_b{bound}_{pool_str}_{cell_line}_{postfix}.npz'
    datafile = os.path.join(out_dir, filename)
    np.savez_compressed(datafile, data=data, target=target, inds=inds, compacts=compacts, sizes=sizes)
    print('Saving file:', datafile)
