# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# our implementation of HiCCUPs algorithm.
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import os
import argparse
from HiC_evaluation.utils import *
from data_processing.Read_npz import read_npz
from data_processing.Read_external_norm import read_singlechromosome_norm
from dataset_informations import *
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from scipy.stats import poisson
import json

import warnings
warnings.filterwarnings("ignore")
import logging

def donut_kernel(R2, R1):

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[center[0], :] = 0
    kernel[:, center[1]] = 0

    return kernel

def lowerleft_kernel(R2, R1):

    kernel = np.ones((R1*2+1, R1*2+1))
    center = (R1, R1)
    
    kernel[center[0] - R2 : center[0]+R2+1, center[1] - R2 : center[1]+R2+1] = 0
    kernel[:center[0]+1, :] = 0
    kernel[:, center[1]:] = 0

    return kernel

def horizontal_kernel(R2, R1):

    kernel = np.zeros((3, R1*2+1))
    center = (1, R1)

    kernel[ : , : center[1] - R2 ] = 1
    kernel[ : , center[1] + R2 + 1 : ] = 1
    
    return kernel

def vertical_kernel(R2, R1):

    kernel = np.zeros((R1*2+1, 3))
    center = (R1, 1)

    kernel[ : center[0] - R2, : ] = 1
    kernel[ center[0] + R2 + 1 : , : ] = 1

    return kernel

def get_kernels(peak_size, min_donut_size, max_donut_size):
    return [[donut_kernel(peak_size, d) for d in range(min_donut_size, max_donut_size+1)], 
            [lowerleft_kernel(peak_size, d) for d in range(min_donut_size, max_donut_size+1)], 
            [horizontal_kernel(peak_size, d) for d in range(min_donut_size, max_donut_size+1)], 
            [vertical_kernel(peak_size, d) for d in range(min_donut_size, max_donut_size+1)]]

def loop_clustering(peak_cands, 
                    clustering_boundary,
                    singleton_qvalue,
                    info = True):
    num_cands = len(peak_cands)
    peaks_final = []
    while len(peak_cands) > 0:
        top_peak = max(peak_cands)
        peak_cands.remove(top_peak)
        peaks_cluster = [top_peak[1]]
        centroid = top_peak[1]
        r = 0
        find = True
        while find:
            find = False

            def dis(x, y):
                return np.linalg.norm((x[0]-y[0], x[1]-y[1]))
            
            centroid = np.mean(peaks_cluster, axis = 0)
            r = max([dis(peak, centroid) for peak in peaks_cluster ])
                
            for peak in peak_cands:
                if dis(peak[1], centroid) - r < clustering_boundary:
                    peaks_cluster.append(peak[1])
                    peak_cands.remove(peak)
                    find = True
                    break
        if r>0 or top_peak[2] <= singleton_qvalue:
            peaks_final.append((top_peak[1], centroid, r))
    
    if info:
        print_info(f'Found {len(peaks_final)} peaks from {num_cands} candidates')

    return peaks_final

def find_peaks(full_matrix,
               full_norm, 
               compact_idx, 
               bound, 

               peak_size,
               min_donut_size,
               max_donut_size,
               min_reads,
               lambda_step,
               FDR,
               thresholds,
               gap_filter_range,
               clustering_boundary,
               singleton_qvalue,

               info = True):

    kernels = get_kernels(peak_size, min_donut_size, max_donut_size)

    l = full_matrix.shape[0]
    B = min(bound, l)
    window_size = min(2*B, l)
    upper_triangle = np.triu(np.ones((window_size, window_size)), 0)
    
    expect_vector = get_oe_matrix(full_matrix, bound = window_size, oe=False)
    # expect_vector = [1072.7704, 345.14786, 163.42279, 114.05354, 89.0532, 73.90814, 63.634396, 56.2612, 50.60041, 46.2161, 42.565075, 39.603497, 37.0259, 34.779778, 32.879433, 31.154373, 29.619509, 28.230207, 26.998943, 25.843594, 24.775536, 23.80841, 22.921345, 22.08086, 21.286636, 20.57607, 19.844234, 19.220356, 18.620132, 18.042477, 17.496471, 17.009243, 16.472052, 16.018429, 15.5435095, 15.113119, 14.726029, 14.324624, 13.942324, 13.594511, 13.233154, 12.926416, 12.582154, 12.275603, 11.993311, 11.710655, 11.429957, 11.165966, 10.921215, 10.659773, 10.434648, 10.233285, 9.996523, 9.792909, 9.588372, 9.408576, 9.203315, 9.011642, 8.828952, 8.673231, 8.496363, 8.354844, 8.195825, 8.031441, 7.882967, 7.739243, 7.5897727, 7.4662614, 7.3427176, 7.21296, 7.083622, 6.9473248, 6.8341064, 6.705486, 6.609285, 6.513979, 6.3858905, 6.2864933, 6.198932, 6.1039743, 6.007169, 5.926597, 5.8242474, 5.740907, 5.65567, 5.5661907, 5.4818854, 5.394281, 5.3128853, 5.2424645, 5.155781, 5.082538, 5.007289, 4.948369, 4.867192, 4.809555, 4.7380257, 4.672647, 4.6386986, 4.560361, 4.4952664, 4.4872727, 4.384574, 4.3329597, 4.275195, 4.217363, 4.1540694, 4.1127205, 4.0650296, 4.0056295, 3.960349, 3.9157846, 3.8647857, 3.8374636, 3.7751074, 3.7225761, 3.6864245, 3.6401966, 3.5976322, 3.563873, 3.528611, 3.490375, 3.4544165, 3.4143314, 3.377126, 3.341321, 3.305283, 3.2731762, 3.2384133, 3.2032616, 3.168446, 3.1298435, 3.1034954, 3.0731337, 3.0410488, 3.0097544, 2.9810371, 2.9472625, 2.9232905, 2.9041052, 2.8827448, 2.8472133, 2.8163307, 2.791125, 2.7608688, 2.7532084, 2.7124696, 2.6940029, 2.6624699, 2.6467068, 2.605535, 2.5901887, 2.5616784, 2.539283, 2.5191514, 2.5225813, 2.4913468, 2.4606285, 2.4555242, 2.41756, 2.3875906, 2.3689673, 2.352274, 2.3286476, 2.3104746, 2.2884068, 2.2698417, 2.2577515, 2.2509148, 2.209811, 2.2014186, 2.1868455, 2.1651714, 2.1595557, 2.1336691, 2.1187148, 2.0947552, 2.0924149, 2.0702248, 2.0481765, 2.037596, 2.02354, 2.0087714, 1.9958637, 1.9793357, 1.9597468, 1.9523922, 1.9370049, 1.9227432, 1.9080846, 1.900822, 1.8808329, 1.8693691, 1.8583411, 1.8600981, 1.8423698, 1.8210781, 1.8149607, 1.798962, 1.7865561, ] + [0]*200
    # expect_vector = [1395.0327, 480.25955, 230.81374, 161.30719, 126.04364, 104.58276, 90.20132, 79.81479, 71.801834, 65.636375, 60.54796, 56.235466, 52.644115, 49.514782, 46.733986, 44.31635, 42.15827, 40.153515, 38.382477, 36.73185, 35.25452, 33.953648, 32.770653, 31.466417, 30.423777, 29.474905, 28.330618, 27.428146, 26.545277, 25.754955, 25.011795, 24.228762, 23.472683, 22.810629, 22.184443, 21.546482, 20.985264, 20.434998, 19.96255, 19.426714, 18.902864, 18.41408, 17.930908, 17.522741, 17.10385, 16.740364, 16.327349, 15.933978, 15.581148, 15.235983, 14.925806, 14.598253, 14.32105, 14.033697, 13.726377, 13.448204, 13.154685, 12.898959, 12.597552, 12.398986, 12.158519, 11.92517, 11.6795435, 11.455171, 11.25235, 11.050209, 10.855308, 10.672015, 10.489244, 10.309567, 10.124795, 9.932204, 9.795368, 9.611523, 9.499482, 9.3100195, 9.139901, 8.988646, 8.873427, 8.728534, 8.608904, 8.493425, 8.355321, 8.209414, 8.101938, 7.965405, 7.852264, 7.7634854, 7.6593027, 7.5181756, 7.4191008, 7.279593, 7.180503, 7.1107125, 7.0113072, 6.894828, 6.808061, 6.703694, 6.641722, 6.5539565, 6.551673, 6.562457, 6.3076, 6.207734, 6.124075, 6.0308566, 5.95065, 5.8858976, 5.8094516, 5.731282, 5.669241, 5.603151, 5.5374093, 5.4743633, 5.4106894, 5.3432384, 5.287181, 5.2388906, 5.173926, 5.114225, 5.0609546, 5.01043, 4.986988, 4.910524, 4.8469424, 4.7863183, 4.7514124, 4.6881657, 4.6305976, 4.6003213, 4.5478206, 4.4999094, 4.449731, 4.412436, 4.365217, 4.3291097, 4.2694135, 4.2426076, 4.1875367, 4.162319, 4.1386137, 4.0722055, 4.043101, 4.006667, 3.9756484, 3.946989, 3.9100068, 3.8567557, 3.8201072, 3.7934217, 3.7580266, 3.7230165, 3.6858277, 3.651371, 3.6702716, 3.616955, 3.5757346, 3.5292861, 3.5010307, 3.4839935, 3.4433513, 3.420298, 3.378445, 3.3608782, 3.3272128, 3.3012667, 3.277359, 3.241976, 3.2248743, 3.1911407, 3.1858542, 3.152973, 3.1220932, 3.1073418, 3.0728338, 3.0560858, 3.0211394, 3.0025234, 2.9899783, 2.9703171, 2.93978, 2.9155796, 2.9059901, 2.8890374, 2.867216, 2.8395538, 2.8191478, 2.8041487, 2.7827418, 2.7595441, 2.741092, 2.723751, 2.7101266, 2.685587, 2.6907809, 2.6667693, 2.644472, 2.6452405, 2.6142087, 2.585957] + [0]*200
    # expect_vector = [1063.1499, 356.4345, 171.0452, 119.70009, 93.74488, 78.02195, 67.37607, 59.674255, 53.784126, 49.126, 45.333126, 42.170536, 39.439316, 37.073143, 35.00424, 33.15751, 31.496716, 29.995134, 28.655779, 27.408796, 26.25716, 25.190386, 24.204782, 23.268112, 22.431313, 21.643656, 20.867226, 20.173326, 19.51273, 18.877132, 18.268023, 17.6939, 17.13294, 16.622662, 16.111246, 15.655877, 15.214871, 14.783466, 14.363612, 13.975707, 13.612304, 13.25938, 12.894315, 12.561005, 12.242672, 11.949868, 11.650532, 11.363496, 11.093943, 10.822543, 10.576024, 10.336872, 10.09842, 9.893904, 9.670609, 9.464185, 9.259837, 9.054059, 8.861694, 8.687221, 8.507504, 8.345892, 8.164531, 8.005239, 7.8550096, 7.694172, 7.5449266, 7.411493, 7.273555, 7.1343617, 6.999823, 6.8693056, 6.7409496, 6.61809, 6.516336, 6.412812, 6.2842793, 6.1684093, 6.0841475, 5.9817147, 5.8818774, 5.784782, 5.7021775, 5.604273, 5.5110846, 5.428686, 5.3375688, 5.2507725, 5.1720004, 5.0987635, 5.0177255, 4.9340715, 4.8595324, 4.795465, 4.724264, 4.6520596, 4.5889573, 4.5234165, 4.467981, 4.401667, 4.34153, 4.2983413, 4.230711, 4.174038, 4.1223383, 4.0663276, 4.0091653, 3.9618564, 3.9108477, 3.8557868, 3.8057427, 3.7663648, 3.7124026, 3.669057, 3.6195474, 3.5731697, 3.536739, 3.494205, 3.448504, 3.4148536, 3.3778644, 3.340903, 3.3043423, 3.2676876, 3.2319787, 3.1947575, 3.1598463, 3.1244185, 3.0856562, 3.0600898, 3.025418, 2.9812496, 2.9509768, 2.926371, 2.8984714, 2.8648646, 2.8374574, 2.8073447, 2.778095, 2.7543182, 2.7310326, 2.6969051, 2.6751506, 2.6501105, 2.6213908, 2.5956373, 2.5712802, 2.5428495, 2.5275717, 2.498977, 2.473353, 2.4527767, 2.4307382, 2.4032433, 2.3908012, 2.374495, 2.3534923, 2.3285384, 2.3056583, 2.284883, 2.2638528, 2.243278, 2.230262, 2.2100208, 2.1885266, 2.170541, 2.1516728, 2.1368577, 2.1235623, 2.0996678, 2.084672, 2.0670807, 2.053514, 2.0347962, 2.0185702, 2.0082834, 1.9869128, 1.9723213, 1.9562076, 1.9391123, 1.9268156, 1.9109327, 1.8983598, 1.886904, 1.8707318, 1.8565438, 1.8415234, 1.829158, 1.8144801, 1.8017089, 1.7943808, 1.7759938, 1.7618144, 1.7542397, 1.7430077, 1.7251829, 1.7176716, 1.7070096, 1.6920722, 1.682772, ] + [0]*200
    expect = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            if abs(i-j) < len(expect_vector):
                expect[i][j] = expect_vector[abs(i-j)]
    esums = []
    for kernel in kernels:
        esum = []
        for k in kernel:
            esum_d = correlate(expect, k, mode='constant')
            esum.append(esum_d)

        esums.append(esum)

    enriched_pixels = []

    if info:
        pbar = tqdm(range(0, l, B))
    else:
        pbar = range(0, l, B)
    for s0 in pbar:
        if info:
            pbar.set_description(f'Currently find {len(enriched_pixels)} enriched pixels')
            
        s = min(s0, l-window_size)
        matrix = full_matrix[s:s+window_size, s:s+window_size]
        norm   = full_norm  [s:s+window_size]

        norm_mat = np.outer(norm, norm)

        observed = matrix * norm_mat

        observed = (np.rint(observed)).astype(int)

        log_lambda_step = np.log(lambda_step)

        pixel_scores = {}

        # print(observed)

        for kid, kernel in enumerate(kernels):
            msum = correlate(matrix, kernel[0], mode='constant')
            esum = esums[kid][0]

            for d in range(max_donut_size - min_donut_size):
                ll_msum = correlate(matrix, kernels[1][d], mode='constant')
                p = np.where(ll_msum<min_reads & (matrix > 1))
                if len(p[0])>0:
                    msum1 = correlate(matrix, kernel[d+1], mode='constant')
                    esum1 = esums[kid][d+1]
                    msum[p] = msum1[p]
                    esum[p] = esum1[p]
                else:
                    break

            Ek = np.nan_to_num(msum/esum*expect)
            Ek = Ek * norm_mat
            
            #lambda-chunk FDR

            logEk = np.nan_to_num(np.log(Ek))

            bin_id = np.maximum(0, np.ceil(logEk/log_lambda_step).astype(int))
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

                    # if i==171 and j==183:
                    # if i==155 and j==164:
                    # if i==234 and j==354:
                    #     print(pvalue, qvalue, observed[i][j]/Ek[i][j],  observed[i][j], Ek[i][j])
                    #     print(msum[i][j], esum[i][j],msum[i][j]/esum[i][j]*expect[i][j] )

                        # print(expect[i-5:i+6, j-5:j+6])
                
                    if qvalue <= FDR and observed[i][j]/Ek[i][j] > thresholds[kid]:
                        # pass BHFDR, check ratio
                        flag = (kid in (0,1)) and (observed[i][j]/Ek[i][j] > thresholds[-1])
                        if (i,j) not in pixel_scores: pixel_scores[(i,j)] = [0, 0]
                        pixel_scores[(i,j)][0] += 2 + (1 if flag else 0)
                        pixel_scores[(i,j)][1] += qvalue
                
        for p, v in pixel_scores.items():
            if v[0]>=9 and abs(p[0]-p[1]) <= bound:
                enriched_pixels.append((observed[p[0], p[1]], (p[0]+s, p[1]+s), v[1]))
        
    gaps = set(range(l)) - set(compact_idx)
    near_gap = [False for _ in range(l)]
    for gap in gaps:
        for i in range(gap_filter_range):
            if gap-i >= 0: 
                near_gap[gap-i] = True
            if gap+i < l:
                near_gap[gap+i] = True
    
    filtered_enriched_pixels = []
    for pixels in enriched_pixels:
        if not near_gap[pixels[1][0]] and not near_gap[pixels[1][1]] and abs(pixels[1][0] - pixels[1][1]) > peak_size + 2:
            filtered_enriched_pixels.append(pixels)

    peaks_final = loop_clustering(filtered_enriched_pixels, clustering_boundary, singleton_qvalue, info=info)

    return peaks_final

def draw_peaks(matrix, peaks, start1, end1, start2, end2, file_name , vmax  = None):
    plt.clf()
    if vmax:
        plt.imshow(matrix[start1:end1, start2:end2], vmin=0, vmax = vmax, cmap='OrRd')
    else:
        plt.imshow(matrix[start1:end1, start2:end2], vmin=0, cmap='OrRd')

    plt.colorbar()
    peak_top = []
    peak_center = []
    peak_size = []
    for p, c, r in peaks:
        if start1 <= p[0] and p[0] < end1 and start2 <= p[1] and p[1] < end2:
            peak_top.append((p[0]-start1, p[1]-start2))
            peak_center.append((c[0]-start1, c[1]-start2))
            peak_size.append(r+20)

    if len(peak_top)>0:
        peak_top = np.array(peak_top)
        peak_center = np.array(peak_center)
        plt.scatter(peak_top[:, 1], peak_top[:, 0], s = 20, linewidths = 1, facecolors='none', edgecolors='blue')
        plt.scatter(peak_center[:, 1], peak_center[:, 0], s = peak_size, linewidths = 0.5, facecolors='none', edgecolors='blue')
    
    plt.savefig(file_name)

def write_bedpe_annotation(chr, peaks, file, resolution):
    for top, center, radius in peaks:
        x1 = int(top[0] * resolution)
        x2 = int(top[0] * resolution + resolution)
        y1 = int(top[1] * resolution)
        y2 = int(top[1] * resolution + resolution)
        file.write(f'{chr}\t{x1}\t{x2}\t{chr}\t{y1}\t{y2}\t.\t.\t.\t.\t0,0,255\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, required = True)
    parser.add_argument('--hic-caption', type=str, default = 'hic')
    parser.add_argument('--external-norm-file', type=str, 
                        default = f'{root_dir}/{RAW_dir}/#(CELLLINE)/10kb_resolution_intrachromosomal/#(CHR)/MAPQGE30/#(CHR)_10kb.KRnorm')
    parser.add_argument('--resolution', type=str, default='10kb')

    parser.add_argument('--bound', type=int, default=200)
    parser.add_argument('--multiple', type=int, default=255)

    parser.add_argument('-c', '--cell-line', default='GM12878')
    parser.add_argument('-s', dest='dataset', default='test', choices=set_dict.keys(), )

    parser.add_argument('--peak-size', type=int, default = 2)
    parser.add_argument('--min-donut-size', type=int, default = 5)
    parser.add_argument('--max-donut-size', type=int, default = 5)
    parser.add_argument('--min-reads', type=int, default = 16)
    parser.add_argument('--lambda-step', type=float, default=2**(1/3))
    parser.add_argument('--FDR', type=float, default=0.1)
    parser.add_argument('--clustering-boundary', type=float, default=2)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1.75, 1.75, 1.5, 1.5, 2])
    parser.add_argument('--gap-filter-range', type=int, default=5)
    parser.add_argument('--singleton-qvalue', type=float, default=0.02)

    args = parser.parse_args()

    data_dir = args.data_dir
    hic_caption = args.hic_caption
    external_norm_file = args.external_norm_file
    res = args.resolution
    bound = args.bound
    multiple = args.multiple

    cell_line = args.cell_line
    dataset = args.dataset
    
    resolution = res_map[res.split('_')[0]]

    peak_size = args.peak_size
    min_donut_size = args.min_donut_size
    max_donut_size = args.max_donut_size
    min_reads = args.min_reads
    lambda_step = args.lambda_step
    FDR = args.FDR
    thresholds = args.thresholds
    gap_filter_range = args.gap_filter_range
    clustering_boundary = args.clustering_boundary
    singleton_qvalue = args.singleton_qvalue

    chr_list = set_dict[dataset]
    abandon_chromosome = abandon_chromosome_dict.get(cell_line, [])

    save_dir = os.path.join(data_dir, res, 'HiCCUPS')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log_file_path = os.path.join(save_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file_path, 
        filemode='w', 
    )

    loop_counts = []
    annotation = open(os.path.join(save_dir, 'HiCCUPS_loop_annotation.bedpe'), 'w')
    annotation.write(f'chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tname\tscore\tstrand1\tstrand2\tcolor\n')
    for n in chr_list:
        if n in abandon_chromosome:
            continue

        in_file = os.path.join(data_dir, f'chr{n}_{res}.npz')
        
        matrix, compact_idx, norm = read_npz(in_file, hic_caption=hic_caption, bound = bound, multiple=multiple, include_additional_channels=False)

        if external_norm_file != 'NONE':
            if '#(CHR)' in external_norm_file:
                norm = read_singlechromosome_norm(external_norm_file, n, cell_line)
            else:
                raise NotImplementedError

        peaks = find_peaks(matrix, 
                           norm, 
                           compact_idx, 
                           bound,
                           peak_size,
                           min_donut_size,
                           max_donut_size,
                           min_reads,
                           lambda_step,
                           FDR,
                           thresholds,
                           gap_filter_range,
                           clustering_boundary,
                           singleton_qvalue)

        # in_file = os.path.join(data_dir, f'chr{n}_{"10kb"}.npz')
        
        # matrix, compact_idx, norm = read_npz(in_file, hic_caption=hic_caption, bound = bound, multiple=multiple, include_additional_channels=False)

        # w = 20
        # start1, end1 = 50 -w, 50 +w
        # start2, end2 = 250-w, 250+w
        # oe,_ = get_oe_matrix(matrix, bound=200)       
        # draw_peaks(oe, peaks, start1 = start1, end1 = end1, start2 = start2, end2 = end2, file_name=os.path.join(save_dir, f'chr{n}_{start1}to{end1}_{start2}to{end2}_oe.pdf'))
        # draw_peaks(matrix, peaks, start1 = start1, end1 = end1, start2 = start2, end2 = end2, file_name=os.path.join(save_dir, f'chr{n}_{start1}to{end1}_{start2}to{end2}.pdf'))

        loop_counts.append(len(peaks))
        write_bedpe_annotation('chr'+str(n), peaks, annotation, resolution)

        # break

    print_info(f'Chromosome loop counts: {loop_counts}')
    print_info(f'Total loop counts: {np.sum(loop_counts)}')
