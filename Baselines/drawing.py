import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numba import jit
from Arg_Parser import root_dir
import argparse

# You can use the following functions to draw heatmap and triangular heatmap
# of Hi-C matrices.
def heatmap2d(arr: np.ndarray, save_name=None):
    """
    plt heatmap of 2d array
    """
    plt.imshow(arr, cmap='OrRd')
    plt.colorbar()
    if save_name:
      plt.savefig(save_name)
    else:
      plt.show()
    
def heatmap2d_bound(arr: np.ndarray, vmin=0, vmax=1, save_name=None):
    """
    plt heatmap of 2d array with bound
    """
    plt.imshow(arr, cmap='OrRd', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if save_name:
      plt.savefig(save_name)
    else:
      plt.show()
    
def heatmap2d_invert(arr: np.ndarray, vmin=0, vmax=1, save_name=None):
    """
    plt heatmap of reversed 2d array
    """
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(arr, cmap='OrRd', vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()
    cax = fig.add_axes([ax.get_position().x1+0.015,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    
    if save_name:
      plt.savefig(save_name)
    else:
      plt.show()
    
def draw_triangular_map(arr: np.ndarray, height=500, save_name="triangular_map.pdf"):
  """
  plt triangular heatmap of 2d array with height
  """
  converted_matrix = np.zeros_like(arr)
  for i in range(converted_matrix.shape[0] // 2):
    for j in range(0, i):
      converted_matrix[i, j] = np.nan
    for j in range(i, converted_matrix.shape[0] - i):
      converted_matrix[i, j] = arr[i + j, j - i]
    for j in range(converted_matrix.shape[0] - i, converted_matrix.shape[0]):
      converted_matrix[i, j] = np.nan
  tgt_matrix = converted_matrix[:height, :]
  
  heatmap2d_invert(tgt_matrix, save_name=save_name)
  

@jit(nopython=True)
def get_new_matrix(matrix, new_matrix, weight_factor):
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[0]):
      if matrix[i][j] != 0:
        new_matrix[i][j] = matrix[i][j] / weight_factor[abs(i-j)]
  return new_matrix

@jit(nopython=True)
def get_new_matrix_with_bound(matrix, new_matrix, weight_factor, boundary):
  for i in range(matrix.shape[0]):
      for j in range(max(i - boundary + 1, 0), min(matrix.shape[0], i + boundary), 1):
        if matrix[i][j] != 0:
          new_matrix[i][j] = matrix[i][j] / weight_factor[abs(i-j)]
  return new_matrix

@jit(nopython=True)
def compute_factor(matrix, weight_list, weight_count_all):
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[0]):
        if matrix[i][j] != 0:
          weight_list[abs(i-j)] += matrix[i][j]
    weight_factor = np.array(weight_list) / np.array(weight_count_all)
    return weight_factor
   
def get_oe_matrix(matrix, boundary=None):
  matrix_len = matrix.shape[0]
  if not boundary:
    weight_list = [0. for i in range(matrix_len)]
    weight_count_all = [(matrix_len - i) * 2 for i in range(matrix_len)]
    for i in tqdm(range(matrix.shape[0])):
      weight_list[i] = np.trace(matrix, offset=i) + np.trace(matrix, offset=-i)
    weight_factor = np.array(weight_list) / np.array(weight_count_all)
    new_matrix = np.zeros_like(matrix, dtype=np.float32)
    new_matrix = get_new_matrix(matrix, new_matrix, weight_factor)
  else:
    weight_list = [0. for i in range(boundary)]
    weight_count_all = [(matrix_len - i) * 2 for i in range(boundary)]
    for i in tqdm(range(boundary)):
      weight_list[i] = np.trace(matrix, offset=i) + np.trace(matrix, offset=-i)
    weight_factor = np.array(weight_list) / np.array(weight_count_all)
    new_matrix = np.zeros_like(matrix, dtype=np.float32)
    new_matrix = get_new_matrix_with_bound(matrix, new_matrix, weight_factor, boundary)
  return new_matrix


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='HiCARN_1_GM12878_total')
  parser.add_argument('--chr', type=str, default=1)
  args = parser.parse_args()
  args = args.__dict__
  chr = args['chr']
  result_dir = f"{root_dir}/predict/{args['name']}"
  a = np.load(result_dir+f"/predict_chr{chr}_40kb.npz")['hic']
  b = get_oe_matrix(a, boundary=200)
  c = np.corrcoef(b)
  draw_triangular_map(c, height=300)
  