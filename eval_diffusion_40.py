# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the HiCARN implementation, a script to predict HR matrices with diffusion-based approaches.
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------

import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
from Arg_Parser import *
from utils import *
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torch import nn
import json
import random

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
    info_str = filename.split('.')[0].split('_')
    chunk = get_digit(info_str[4])
    stride = get_digit(info_str[5])
    diagonal_stride = get_digit(info_str[6])
    bound = get_digit(info_str[7])
    scale = 1 if info_str[8] == 'nonpool' else get_digit(info_str[8])
    return chunk, stride, diagonal_stride, bound, scale

from imagen_pytorch.imagen_pytorch import GaussianDiffusionContinuousTimes, log_snr_to_alpha_sigma, right_pad_dims_to, default, log
from functools import partial

class DDIM_noisescheduler(GaussianDiffusionContinuousTimes):
    def __init__(self, *, noise_schedule, timesteps=1000, eta = 0):
        super().__init__(noise_schedule=noise_schedule, timesteps=timesteps)

        self.eta = eta
    
    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        posterior_sigma = self.eta * (sigma_next/sigma) * (1-alpha**2/alpha_next**2).sqrt()

        adjusted_sigma_next = (sigma_next**2 - posterior_sigma**2).sqrt()

        posterior_mean = alpha_next * x_start + adjusted_sigma_next * x_t

        posterior_variance = posterior_sigma**2
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

def predictor(loader, ckpt_file, diffusion_steps, device, data_file):
    args_file = os.path.join(os.path.dirname(ckpt_file), 'args.json')
    with open(args_file) as f:
        args = json.load(f)

    with open(args['weights_file']) as f:
        weights = json.load(f)
    
    channel_weights = np.expand_dims(weights[args['dataset']], (0, 2, 3))
    channel_weights = torch.from_numpy(channel_weights).to(device).float()

    def data_normalize(data):
        if len(args['reverse_channels']) > 0 :
            data[:, args['reverse_channels'], :, :] = 1 - data[:, args['reverse_channels'], :, :]
        data = data * channel_weights
        return data

    unet1 = Unet(
        cond_on_text = False,
        dim = args['unet_input_dim'],
        channels = len(args['output_channels']),
        cond_images_channels = len(args['input_channels']),
        dim_mults = (1, 2, 4),
        num_resnet_blocks = args['unet_resnet_depth'],
        layer_attns = (False, True, True),
        layer_cross_attns = False,
    )

    imagen = Imagen(
        condition_on_text = False,
        channels = len(args['output_channels']),
        unets = (unet1,),
        image_sizes = (chunk,),
        timesteps = args['diffusion_steps'],
        pred_objectives = 'noise',
    ).cuda()

    trainer = ImagenTrainer(imagen, cosine_decay_max_steps=args['cosine_decay_max_steps'])

    trainer.load(ckpt_file)

    trainer.to(device)

    trainer.imagen.noise_schedulers = nn.ModuleList([GaussianDiffusionContinuousTimes(noise_schedule='linear', timesteps=diffusion_steps)]).to(device)
    
    result_data = []
    result_inds = []
            
    chr_nums = get_chr_nums(data_file)
    
    results_dict = dict()
    test_metrics = dict()
    for chr in chr_nums:
        test_metrics[f'{chr}'] = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
        results_dict[f'{chr}'] = [[], [], [], []]  # Make respective lists for ssim, psnr, mse, and repro   
    
    trainer.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='HiCARN Predicting: '):
            lr, hr, inds = batch
            batch_size = lr.size(0)
            ind = f'{(inds[0][0]).item()}'
            test_metrics[ind]['nsamples'] += batch_size
            
            lr = lr.to(device)            
            
            lr = data_normalize(lr)
            
            lr = lr[:, args['input_channels'], :, :]

            out = trainer.sample(cond_images=lr, batch_size=batch_size, use_tqdm=False)
                
            out /= channel_weights[:, args['output_channels'], :, :]

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())

            
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
        
    hics = together(result_data, result_inds, tag='Reconstructing: ')
    
    return hics

def save_data(hic, compact, size, file):
    hic = spreadM(hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hic, compact=compact)
    print('Saving file:', file)


if __name__ == '__main__':
    parser = data_predict_parser()
    parser.add_argument('--diffusion-steps', type=int, default=5, help='number of diffusion steps to use while evaluating')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root-dir', type=str, default='/data/hic_data')
    args = parser.parse_args(sys.argv[1:])
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    cuda = args.cuda
    model = args.model
    steps = args.diffusion_steps
    data_file = args.file_name
    root_dir = args.root_dir

    in_dir = os.path.join(root_dir, 'data')
    out_dir = os.path.join(ckpt_file.split('.')[0], 'predict', cell_line)
    mkdir(out_dir)

    chunk, stride, diagonal_stride, bound, scale = filename_parser(data_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA available? ", torch.cuda.is_available())
    print("Device being used: ", device)

    print(f'Loading data: {data_file}')
    data = np.load(os.path.join(in_dir, data_file), allow_pickle=True)
    loader = dataloader(data)

    indices, compacts, sizes = data_info(data)

    hics = predictor(loader, ckpt_file, steps, device, data)

    def save_data_n(key):
        file = os.path.join(out_dir, f'chr{key}_{"10kb"}.npz')
        save_data(hics[key], compacts[key], sizes[key], file)

    for key in compacts.keys():
        save_data_n(key)
