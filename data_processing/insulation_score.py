# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# compute insulation score as extra channel.
# --------------------------------------------------------

import numpy as np

class insulation_score:
    def __init__(self, distance_lower_bound = 0, distance_upper_bound = 100000,  cutoff = 255, original = True, normalize = True) -> None:
        self.distance_lower_bound = distance_lower_bound
        self.distance_upper_bound = distance_upper_bound
        self.cutoff = cutoff
        self.original = original
        self.normalize = normalize
    def __call__(self, data):
        matrix = data['hic']
        n, _ = matrix.shape
        pref_sum = np.zeros_like(matrix)
        insulation_score = np.zeros_like(matrix, dtype=float)

        for i in range(n-1, -1 ,-1):
            for j in range(i+1, min(i+self.distance_upper_bound, n)):
                pref_sum[i,j] = (pref_sum[i+1, j] if i+1 < n else 0) + (pref_sum[i,j-1] if j > 0 else 0) - (pref_sum[i+1, j-1] if i+1 < n and j > 0 else 0) + matrix[i, j]
                w_size = (j-i)/2
                if abs(i-j)>=self.distance_lower_bound:
                    insulation_score[i,j] = (pref_sum[i,j] - pref_sum[i, i+int(np.floor(w_size))] - pref_sum[j-int(np.floor(w_size)), j])
                    insulation_score[i,j] /= np.square(np.ceil(w_size))
                    insulation_score[j,i] = insulation_score[i,j]

        if self.normalize:
            insulation_score_norm = insulation_score.copy()
            for d in range(self.distance_lower_bound, min(self.distance_upper_bound, n)):
                p = np.arange(n-d)
                e = np.max(insulation_score[p, p+d])
                if e>0:
                    insulation_score_norm[p, p+d] /= e
                    # input matrix should be symmetric
                    insulation_score_norm[p+d, p] /= e

        if self.original:
            insulation_score = np.minimum(insulation_score, self.cutoff)
            insulation_score = insulation_score / self.cutoff
            insulation_score = np.expand_dims(insulation_score, 0)
       
        insulation_score_norm = np.expand_dims(insulation_score_norm, 0)

        if self.original and self.normalize:
            out = np.concatenate([insulation_score, insulation_score_norm], axis = 0)
        elif self.original:
            out = insulation_score
        elif self.normalize:
            out = insulation_score_norm
        else:
            raise NotImplementedError
        return out
