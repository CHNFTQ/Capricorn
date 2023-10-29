# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix

except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}


def readcoo2mat(cooFile, normFile, resolution):
    """
    Function used for reading a coordinated tag file to a square matrix.
    """
    norm = open(normFile, 'r').readlines()
    norm = np.array(list(map(float, norm)))
    compact_idx = list(np.where(np.isnan(norm) ^ True)[0])
    pd_mat = pd.read_csv(cooFile, sep='\t', header=None, dtype=int)
    row = pd_mat[0].values // resolution
    col = pd_mat[1].values // resolution
    val = pd_mat[2].values
    mat = coo_matrix((val, (row, col)), shape=(len(norm), len(norm))).toarray()
    mat = mat.astype(float)
    norm[np.isnan(norm)] = 1
    mat = mat / norm
    mat = mat.T / norm
    HiC = mat + np.tril(mat, -1).T
    return HiC.astype(int), compact_idx


def compactM(matrix, compact_idx, verbose=False):
    """
    Compacts the matrix according to the index list.
    """
    compact_size = len(compact_idx)
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    return result


def spreadM(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """
    Spreads the matrix according to the index list (a reversed operation to compactM).
    """
    result = np.zeros((full_size, full_size)).astype(c_mat.dtype)
    if convert_int: result = result.astype(np.int)
    if verbose: print('Spreading a', c_mat.shape, 'shaped matrix to', result.shape, 'shaped!')
    for i, s_idx in enumerate(compact_idx):
        result[s_idx, compact_idx] = c_mat[i]
    return result


def spreadMdict(mats, compacts, sizes, convert_int=True, verbose=False):
    results = {}
    for key in mats.keys():
        results[key] = spreadM(mats[key], compacts[key], sizes[key], convert_int=convert_int, verbose=verbose)
    return results


def dense2tag(matrix):
    """
    Converts a square matrix (dense) to coo-based tag matrix.
    """
    matrix = np.triu(matrix)
    tag_len = np.sum(matrix)
    tag_mat = np.zeros((tag_len, 2), dtype=np.int)
    coo_mat = coo_matrix(matrix)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data
    start_idx = 0
    for i in range(len(row)):
        end_idx = start_idx + data[i]
        tag_mat[start_idx:end_idx, :] = (row[i], col[i])
        start_idx = end_idx
    return tag_mat, tag_len


def tag2dense(tag, nsize):
    """
    Coverts a coo-based tag matrix to densed square matrix.
    """
    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    dense_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize)).toarray()
    dense_mat = dense_mat + np.triu(dense_mat, k=1).T
    return dense_mat


def downsampling(matrix, down_ratio, verbose=False):
    """
    Downsampling method.
    """
    if verbose: print(f"[Downsampling] Matrix shape is {matrix.shape}")
    tag_mat, tag_len = dense2tag(matrix)
    sample_idx = np.random.choice(tag_len, tag_len // down_ratio)
    sample_tag = tag_mat[sample_idx]
    if verbose: print(f'[Downsampling] Sampling 1/{down_ratio} of {tag_len} reads')
    down_mat = tag2dense(sample_tag, matrix.shape[0])
    return down_mat


def divide(mat, chr_num, chunk_size=40, stride=28, bound=201, padding=True, species='hsa', verbose=False):
    """
    Dividing method.
    """
    chr_str = str(chr_num)
    if isinstance(chr_num, str): chr_num = except_chr[species][chr_num]
    result = []
    index = []
    size = mat.shape[0]
    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i - j) <= bound and i + chunk_size < height and j + chunk_size < width:
                subImage = mat[i:i + chunk_size, j:j + chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose: print(
        f'[Chr{chr_str}] Deviding HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, '
        f'stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index


def together(matlist, indices, corp=0, species='hsa', tag='HiC'):
    """
    Constructs a full dense matrix.
    """
    chr_nums = sorted(list(np.unique(indices[:, 0])))
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]: chr_nums[-1] = except_chr[species][chr_nums[-1]]
    print(f'{tag} data contain {chr_nums} chromosomes')
    _, h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:, 0] == num)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0, 1]
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            try:
                full_mat[i:i + h, j:j + w] = sub
            except:
                tmp_h, tmp_w = full_mat[i:i + h, j:j + w].shape
                full_mat[i:i + h, j:j + w] = sub[:,:tmp_h, :tmp_w]
        results[n] = full_mat
    return results


def pooling(mat, scale, pool_type='max', return_array=False, verbose=True):
    mat = torch.tensor(mat).float()
    if len(mat.shape) == 2:
        mat.unsqueeze_(0)  # need to add channel dimension
    if pool_type == 'avg':
        out = F.avg_pool2d(mat, scale)
    elif pool_type == 'max':
        out = F.max_pool2d(mat, scale)
    if return_array:
        out = out.squeeze().numpy()
    if verbose:
        print('({}, {}) sized matrix is {} pooled to ({}, {}) size, with {}x{} down scale.'.format(*mat.shape[-2:],
                                                                                                   pool_type,
                                                                                                   *out.shape[-2:],
                                                                                                   scale, scale))
    return out


def dense2sparse(mat, key, low_range, up_range):
    """
    Convert npz file to sparse in format: chr1 bin1 chr2 bin2 value. Mat should be an npz file, the key is 'hicarn'
    for predicted images or 'hic' for real/downsampled images. Chromosome is an integer representing the chromosome of
    the matrix being converted.

    :arg: Matrix = (nxn), Key = ('hicarn' or 'hic'), Chromosome = integer

    :returns: List of values for each bin for one chromosome in the form [chromosome, bin1, chromosome, bin2, value]

    """

    x = np.load(mat)
    if key == "hic":
        y = np.array(x['hic'])
    elif key == 'hicarn':
        y = np.array(x['hicarn'])

    z = y[low_range:up_range, low_range:up_range]

    height, width = z.shape
    assert height == width, ' Assumed the matrix is square.'

    final_list = list()
    for i in range(0, height):
        for j in range(0, width):
            value = z[i, j]
            array = [(i + low_range), (j + low_range), value]
            final_list.append(array)
    final_list = np.array(final_list)
    return final_list


def reference_regions(mat, key, chromosome, resolution):
    """
    Iterates through each matrix bin to separate each bin into reference regions of base pairs for a single chromosome.
    Mat should be an npz file, the key is 'hicarn' for predicted images or 'hic' for real/downsampled images.

    :arg: Matrix = (nxn), Key = ('hicarn' or 'hic'), Chromosome = integer, Resolution = integer

    :returns: List of reference regions in [chromosome, bin start (base pairs), bin end (base pairs), bin number]
    """
    x = np.load(mat)
    if key == "hic":
        y = np.array(x['hic'])
    elif key == 'hicarn':
        y = np.array(x['hicarn'])

    num_bins = y.shape[0]

    count = 0
    final_dict = dict()
    for i in range(0, num_bins + 1):
        if i == 0:
            start = count
            count += (resolution)
            end = count

            array = [chromosome, start, end]
            final_dict[i] = array

        elif 0 < i < num_bins:
            start = count
            count += resolution
            end = count

            array = [chromosome, start, end]
            final_dict[i] = array

        elif i == num_bins:
            start = (count)
            count += (resolution)
            end = count

            array = [chromosome, start, end]
            final_dict[i] = array

    return final_dict


def get_region(region_dict, up_range, low_range):
    final_list = list()
    for value in region_dict.values():
        if low_range <= int(value[2]) <= up_range:
            final_list.append(value)

    return final_list
