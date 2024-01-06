# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Using HiCCUPs to compute loop statistics as extra channel.
# --------------------------------------------------------

import numpy as np
from scipy.ndimage import correlate
from scipy.stats import poisson

def donut_kernel(R1, R2):

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[center[0], :] = 0
    kernel[:, center[1]] = 0

    return kernel


def lowerleft_kernel(R1, R2):

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[:center[0]+1, :] = 0
    kernel[:, center[1]:] = 0

    return kernel


def horizontal_kernel(R1, R2):
    
    kernel = np.zeros((3, R1*2+1))
    center = (1, R1)

    kernel[ : , : center[1] - R2 ] = 1
    kernel[ : , center[1] + R2 + 1 : ] = 1
    
    return kernel

def vertical_kernel(R1, R2):

    kernel = np.zeros((R1*2+1, 3))
    center = (R1, 1)

    kernel[ : center[0] - R2, : ] = 1
    kernel[ center[0] + R2 + 1 : , : ] = 1

    return kernel


class HiCCUPS:
    def __init__(self, 
                 donut_size = 5, 
                 peak_size = 2,
                 lambda_step = 2**(1/3),
                 distance_lower_bound = 0, 
                 distance_upper_bound = 100000,  
                 ratio_cutoff = [1.75/0.8, 1.75/0.8, 1.5/0.8, 1.5/0.8, 2/0.8], 
                 eps = 1e-3) -> None:
        self.donut_size = donut_size
        self.peak_size = peak_size
        self.lambda_step = lambda_step
        self.distance_lower_bound = distance_lower_bound
        self.distance_upper_bound = distance_upper_bound
        self.ratio_cutoff = ratio_cutoff
        self.eps = eps
    def __call__(self, data):
        full_matrix = data['hic']
        full_norm = data['norm']
        kernels = [donut_kernel(self.donut_size, self.peak_size), 
                   lowerleft_kernel(self.donut_size, self.peak_size), 
                   horizontal_kernel(self.donut_size, self.peak_size), 
                   vertical_kernel(self.donut_size, self.peak_size)]
        
        l = full_matrix.shape[0]

        B = min(self.distance_upper_bound, l)
        window_size = min(2*B, l)

        expect_vector = []
        for d in range(l):
            expect_vector.append(np.trace(full_matrix, d)/(l-d))

        upper_triangle = np.triu(np.ones((window_size, window_size)), 0)
        
        expect = np.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                if abs(i-j) < len(expect_vector):
                    expect[i][j] = expect_vector[abs(i-j)]
                    
        esums = []
        for kernel in kernels:
            esum = correlate(expect, kernel, mode='constant') + self.eps
            esums.append(esum)
        
        qvalues = np.tile(np.ones_like(full_matrix), (len(kernels), 1,1)).astype(float)
        ratios = np.zeros_like(qvalues).astype(float)

        for s0 in range(0, l, B):
            s = min(s0, l-window_size)
            matrix = full_matrix[s:s+window_size, s:s+window_size]
            norm   = full_norm  [s:s+window_size]

            norm_mat = np.outer(norm, norm)

            observed = matrix * norm_mat

            observed = (np.rint(observed)).astype(int)

            log_lambda_step = np.log(self.lambda_step)

            for kid, kernel in enumerate(kernels):
                msum = correlate(matrix, kernel, mode='constant')
                esum = esums[kid]

                Ek = msum/esum*expect
                Ek = Ek * norm_mat + self.eps

                # print(observed.shape)
                # print(Ek.shape)
                # print(s, window_size)

                ratios[kid, s:s+window_size, s:s+window_size] = observed/Ek
                
                #lambda-chunk FDR

                logEk = np.log(Ek)

                bin_id = np.ceil(np.maximum(0, logEk)/log_lambda_step).astype(int)
                pvalues = poisson.sf(observed, np.exp(bin_id*log_lambda_step))

                max_bin = bin_id.max()+1
                
                for id in range(max_bin):
                    bin_pos = np.where((bin_id == id) & (upper_triangle == 1))
                    p = pvalues[bin_pos]

                    bin = sorted(zip(p.tolist(), bin_pos[0].tolist(), bin_pos[1].tolist()))
                    size = len(bin)

                    qvalue = 1
                    for rank in range(len(bin), 0, -1):
                        pvalue, i, j = bin[rank-1]
                        qvalue = min(qvalue, pvalue /(rank / size))

                        qvalues[kid, i+s, j+s] = qvalues[kid, j+s, i+s] = qvalue
        
        extra_ratio = np.maximum(ratios[0], ratios[1])
        extra_ratio = np.expand_dims(extra_ratio, 0)
        ratios = np.concatenate([ratios, extra_ratio], axis = 0)
        for kid, cutoff in enumerate(self.ratio_cutoff):
            ratios[kid] = np.minimum(ratios[kid], cutoff)
            ratios[kid] /= cutoff
        max_qvalue = np.max(qvalues, axis=0, keepdims=True)
        min_ratio = np.min(ratios, axis=0, keepdims=True)
        out = np.concatenate([max_qvalue, min_ratio], axis = 0)
        return out
