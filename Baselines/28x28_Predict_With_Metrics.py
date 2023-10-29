# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import sys
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from math import log10
from tqdm import tqdm
import Models.HiCSR as HiCSR
import Models.HiCNN as HiCNN
import Models.HiCPlus as HiCPlus
import torch
import torch.nn.functional as F
from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
from Utils.io import spreadM, together

from Arg_Parser import *


# Adjust 40x40 data for HiCSR/HiCNN/HiCPlus 28x28 output
def predict(model, data):
    padded_data = F.pad(data, (6, 6, 6, 6), mode='constant')
    predicted_mat = torch.zeros((1, 1, padded_data.shape[2], padded_data.shape[3]))
    predicted_mat = model(padded_data).to(device)
    return predicted_mat

def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
    
def get_chr_nums(data):
	inds = torch.tensor(data['inds'], dtype=torch.long)
	chr_nums = sorted(list(np.unique(inds[:, 0])))
	return chr_nums


def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale


def hicarn_predictor(model, hicarn_loader, ckpt_file, device, data_file):
    deepmodel = model.Generator().to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
    print(f'Loading HiCARN checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []
    target_data = []
    target_inds = []
    chr_nums = get_chr_nums(data_file)
    
    results_dict = dict()
    test_metrics = dict()
    for chr in chr_nums:
        test_metrics[f'{chr}'] = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
        results_dict[f'{chr}'] = [[], [], [], []]  # Make respective lists for ssim, psnr, mse, and repro   
	
    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(hicarn_loader, desc='HiCARN Predicting: '):
            lr, hr, inds = batch
            batch_size = lr.size(0)
            ind = f'{(inds[0][0]).item()}'
            test_metrics[ind]['nsamples'] += batch_size
            lr = lr.to(device)[:,:1,:,:]#multi
            hr = hr.to(device)[:,:1,:,:]#multi
            out = predict(deepmodel, lr)
            
            batch_mse = ((out - hr) ** 2).mean()
            test_metrics[ind]['mse'] += batch_mse * batch_size
            batch_ssim = ssim(out, hr)
            test_metrics[ind]['ssims'] += batch_ssim * batch_size
            test_metrics[ind]['psnr'] = 10 * log10(1 / (test_metrics[ind]['mse'] / test_metrics[ind]['nsamples']))            
            test_metrics[ind]['ssim'] = test_metrics[ind]['ssims'] / test_metrics[ind]['nsamples']            
            ((results_dict[ind])[0]).append((test_metrics[ind]['ssim']).item())
            ((results_dict[ind])[1]).append(batch_mse.item())
            ((results_dict[ind])[2]).append(test_metrics[ind]['psnr'])
                
            for i, j in zip(hr, out):
                out1 = torch.squeeze(j, dim=0)
                hr1 = torch.squeeze(i, dim=0)
                out2 = out1.cpu().detach().numpy()
                hr2 = hr1.cpu().detach().numpy()
                genomeDISCO = compute_reproducibility(out2, hr2, transition=True)
                ((results_dict[ind])[3]).append(genomeDISCO)
        		
            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
            target_data.append(hr.to('cpu').numpy())
            target_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    target_data = np.concatenate(target_data, axis=0)
    target_inds = np.concatenate(target_inds, axis=0)
    
    mean_ssims = []
    mean_mses = []
    mean_psnrs = []
    mean_gds = []
        
    for key, value in results_dict.items():
        value[0] = round(sum(value[0])/len(value[0]), 10)
        value[1] = round(sum(value[1])/len(value[1]), 10)
        value[2] = round(sum(value[2])/len(value[2]), 10)
        value[3] = round(sum(value[3])/len(value[3]), 10)    
        mean_ssims.append(value[0])
        mean_mses.append(value[1])
        mean_psnrs.append(value[2])
        mean_gds.append(value[3])
        
        print("\n")
        print("Chr", key, "SSIM: ", value[0])
        print("Chr", key, "MSE: ", value[1])
        print("Chr", key, "PSNR: ", value[2])
        print("Chr", key, "GenomeDISCO: ", value[3])

    print("\n")
    print("___________________________________________")
    print("Means across chromosomes")
    print("SSIM: ", round(sum(mean_ssims) / len(mean_ssims), 10))
    print("MSE: ", round(sum(mean_mses) / len(mean_mses), 10))
    print("PSNR: ", round(sum(mean_psnrs) / len(mean_psnrs), 10))
    print("GenomeDISCO: ", round(sum(mean_gds) / len(mean_gds), 10))
    print("___________________________________________")
    print("\n")
    
    hicarn_hics = together(result_data, result_inds, tag='Reconstructing: ')
    target_hics = together(target_data, target_inds, tag='Reconstructing: ')
    return hicarn_hics, target_hics        	

def save_data(hicarn_hic, compact, size, file):
    hicarn = spreadM(hicarn_hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hicarn, compact=compact)
    print('Saving file:', file)


if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    cuda = args.cuda
    model = args.model
    HiCARN_file = args.file_name
    print('WARNING: Prediction process requires a large memory. Ensure that your machine has ~150G of memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()

    in_dir = os.path.join(root_dir, 'data')
    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]

    chunk, stride, bound, scale = filename_parser(HiCARN_file)

    device = torch.device(
        f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data: {HiCARN_file}')
    hicarn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
    hicarn_loader = dataloader(hicarn_data)

    if model == "HiCSR":
        model = HiCSR

    if model == "HiCPlus":
        model = HiCPlus

    if model == "HiCNN":
        model = HiCNN

    indices, compacts, sizes = data_info(hicarn_data)
    hicarn_hics, target_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device, hicarn_data)


    def save_data_n(key):
        file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
        save_data(hicarn_hics[key], compacts[key], sizes[key], file)
        file = os.path.join(out_dir, f'target_chr{key}_{low_res}.npz')
        save_data(target_hics[key], compacts[key], sizes[key], file)


    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = 3 for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
