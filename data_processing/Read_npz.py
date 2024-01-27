# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# a script to help read from npz format hic matrices.
# --------------------------------------------------------

import numpy as np

def read_npz(
        file_name, 
        hic_caption='hic', 
        bound = None,
        multiple = 1,
        include_additional_channels = True,
        compact_idx_caption = 'compact',
        norm_caption = 'norm',
        ):

    data = np.load(file_name)
    if include_additional_channels:
        hic_matrix = data[hic_caption]
        hic_matrix[0] *= multiple
        
    else:
        hic_matrix = data[hic_caption]
        if len(hic_matrix.shape) >= 3:
            hic_matrix = hic_matrix[0]
        
        hic_matrix *= multiple
    
    if bound is not None:
        mask = np.zeros((hic_matrix.shape[-2], hic_matrix.shape[-1]))
        for i in range(mask.shape[0]):
            for j in range(max(i-bound+1, 0), min(i+bound, mask.shape[1])):         
                mask[i][j] = 1
        
        hic_matrix = hic_matrix * mask

    if compact_idx_caption in data:
        compact_idx = data[compact_idx_caption]
    else:
        compact_idx = [i for i in range(hic_matrix.shape[-2])]

    if norm_caption in data:
        norm = data[norm_caption]
    else:
        norm = np.ones(hic_matrix.shape[-2])

    return hic_matrix, compact_idx, norm