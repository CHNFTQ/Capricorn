# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# discover TAD as extra channel.
# --------------------------------------------------------

import numpy as np
from HiC_evaluation.insulation_score import compute_insulation_score, compute_bounds
from utils import compactM, spreadM

class TAD:
    def __init__(self, window_size = 50, delta_smooth_size = 10, bound_strength = 0.1) -> None:
        self.window_size = window_size
        self.delta_smooth_size = delta_smooth_size
        self.bound_strength = bound_strength

    def __call__(self, data):
        matrix = data['hic']
        compact_idx = data['compact']

        full_size = matrix.shape[0] 

        matrix = compactM(matrix, compact_idx)
        
        insulation_score = compute_insulation_score(matrix, self.window_size)
        TAD_bounds = compute_bounds(insulation_score, self.delta_smooth_size, self.bound_strength)
        
        out = np.zeros_like(matrix)
        last_bound = 0
        for bound in TAD_bounds:
            out[last_bound:bound+1, last_bound:bound+1] = 1
            last_bound = bound
        
        out[last_bound:, last_bound:] = 1
        out = spreadM(out, compact_idx, full_size, convert_int=False)
        out = np.expand_dims(out, 0)
        return out
