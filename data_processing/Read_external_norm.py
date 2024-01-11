import numpy as np

def read_singlechromosome_norm(external_norm_file, n, cell_line):
    CHR_norm_File = external_norm_file
    CHR_norm_File = CHR_norm_File.replace('#(CHR)', 'chr'+str(n))
    CHR_norm_File = CHR_norm_File.replace('#(CELLLINE)', cell_line)

    norm = open(CHR_norm_File, 'r').readlines()
    norm = np.array(list(map(float, norm)))
    norm[np.isnan(norm)] = 1

    return norm
