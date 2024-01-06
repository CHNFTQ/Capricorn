# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# utilities for HiC matrix evaluation.
# --------------------------------------------------------

import numpy as np
import logging

def get_oe_matrix(matrix, bound = 100000000, oe=True):
    max_offset = min(matrix.shape[0], bound)
    
    expected = [np.mean(np.diagonal(matrix, offset)) for offset in range(max_offset)]
    
    if oe:
        e_matrix = np.zeros_like(matrix, dtype=np.float32)
        oe_matrix = np.zeros_like(matrix, dtype=np.float32)
        for i in range(matrix.shape[0]):
            for j in range(max(i-bound+1, 0), min(i+bound, matrix.shape[1])):
                e_matrix[i][j] = expected[abs(i-j)]
                oe_matrix[i][j] = matrix[i][j]/expected[abs(i-j)] if expected[abs(i-j)] != 0 else 0

        return oe_matrix, e_matrix
    else:
        return expected


def print_info(s:str):
    print(s)
    logging.info(s)
