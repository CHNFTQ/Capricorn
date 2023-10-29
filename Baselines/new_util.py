# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import numpy as np

# add multichannel support
def compactM(matrix, compact_idx, verbose=False):
    """
    Compacts the matrix according to the index list.
    """
    compact_size = len(compact_idx)
    new_shape = list(matrix.shape)

    new_shape[-1] = compact_size
    new_shape[-2] = compact_size

    result = np.zeros(new_shape).astype(matrix.dtype)
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    for i, idx in enumerate(compact_idx):
        result[..., i, :] = matrix[..., idx, compact_idx]
    return result

# add multichannel support
def spreadM(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """
    Spreads the matrix according to the index list (a reversed operation to compactM).
    """

    new_shape = list(c_mat.shape)

    new_shape[-1] = full_size
    new_shape[-2] = full_size

    result = np.zeros(new_shape).astype(c_mat.dtype)

    if convert_int: result = result.astype(np.int)
    if verbose: print('Spreading a', c_mat.shape, 'shaped matrix to', result.shape, 'shaped!')
    for i, s_idx in enumerate(compact_idx):
        result[..., s_idx, compact_idx] = c_mat[..., i, :]
    return result
