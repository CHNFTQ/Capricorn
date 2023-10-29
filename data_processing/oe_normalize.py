# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# compute O/E normalized matrix as extra channel.
# --------------------------------------------------------

import numpy as np

class oe_normalize:
    def __init__(self, cutoff = 16) -> None:
        self.cutoff = cutoff
    def __call__(self, data):
        matrix = data['hic']
        n, _ = matrix.shape
        out = np.copy(matrix).astype(float)
        for d in range(n):
            p = np.arange(n-d)
            e = np.mean(matrix[p, p+d])
            if e>0:
                out[p, p+d] /= e
                # input matrix should be symmetric
                out[p+d, p] /= e
            
        out = np.minimum(out, self.cutoff)
        out = out / self.cutoff
        out = np.expand_dims(out, 0)
        return out
