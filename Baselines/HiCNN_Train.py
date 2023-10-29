# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# HiCNN: http://dna.cs.miami.edu/HiCNN2/ (in paper: https://www.mdpi.com/2073-4425/10/11/862)
# --------------------------------------------------------
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from Utils.SSIM import ssim
from math import log10
from Arg_Parser import root_dir
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
cs = np.column_stack
import random
def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--name', type=str, default='HiCNN_GM12878_total')
parser.add_argument('--train_name', type=str, default='GM12878_train')
parser.add_argument('--valid_name', type=str, default='GM12878_valid')
parser.add_argument('--save_epoch', type=int, default=99999)
parser.add_argument('--model', type=str, default='HiCNN')
args = parser.parse_args()
args = args.__dict__
if args['model'] == 'HiCNN':
    from Models.HiCNN import Generator
else:
    raise NotImplementedError
def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
# data_dir: directory storing processed data
data_dir = os.path.join(root_dir, 'data')

# out_dir: directory storing checkpoint files
out_dir = os.path.join(root_dir, 'checkpoints')
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
visdom_str = time.strftime('%m%d')

resos = '10kb10kb_d16_seed0'
chunk = 40
stride = 40
bound = 200
ds_train = 40
ds_valid = 40
pool = 'nonpool'
name = args['name']
train_name = args['train_name']
valid_name = args['valid_name']

num_epochs = 500
batch_size = 64

os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

train_file = os.path.join(data_dir, f'Multi_{resos}_c{chunk}_s{stride}_ds{ds_train}_b{bound}_{pool}_{train_name}.npz')
train = np.load(train_file)

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

train_set = TensorDataset(train_data, train_target, train_inds)

valid_file = os.path.join(data_dir, f'Multi_{resos}_c{chunk}_s{stride}_ds{ds_valid}_b{bound}_{pool}_{valid_name}.npz')
valid = np.load(valid_file)

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

# load network

if args['model'] == 'HiCNN':
    netG = Generator().to(device)
else:
    raise NotImplementedError
# loss function
criterionG = nn.MSELoss().to(device)

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)
# scheduler = ReduceLROnPlateau(optimizerG, 'min')
ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []
best_ssim = 0
best_vloss = 999999
for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}
    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)
    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()

    netG.train()
    train_bar = tqdm(train_loader)
    step = 0
    for data, target, _ in train_bar:
        data = data[:,:1,:,:]
        target = target[:,:1,:,:]
        step += 1
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size

        real_img = target.to(device)
        if args['model'] == 'HiCNN':
            z = F.pad(data, (6, 6, 6, 6), mode='constant')
            z = z.to(device)
        else:
            z = data.to(device)
        fake_img = netG(z)

        ######### Train generator #########
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size
        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
    train_gloss = run_result['g_loss'] / run_result['nsamples']

    valid_result = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            val_lr = val_lr[:,:1,:,:]
            val_hr = val_hr[:,:1,:,:]
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            if args['model'] == 'HiCNN':
                lr = F.pad(val_lr, (6, 6, 6, 6), mode='constant')
                lr = lr.to(device)
            else:
                lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = sr
            hr_out = hr
            g_loss = criterionG(sr, hr)

            valid_result['g_loss'] += g_loss.item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (abs(sr - hr)).mean()
            valid_result['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            valid_result['ssims'] += batch_ssim * batch_size
            valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            valid_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

            batch_ssims.append(valid_result['ssim'])
            batch_psnrs.append(valid_result['psnr'])
            batch_mses.append(batch_mse)
            batch_maes.append(batch_mae)
    ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
    psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
    mse_scores.append((sum(batch_mses) / len(batch_mses)))
    mae_scores.append((sum(batch_maes) / len(batch_maes)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']

    if valid_result['g_loss'] < best_vloss:
        best_vloss = valid_result['g_loss']
        print(f'Now, Best vloss is {best_vloss:.6f}')
        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))    

    if epoch % args['save_epoch'] == 0:
        tmp_ckpt_file = f'{datestr}_epoch{epoch}g_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, tmp_ckpt_file))
    # lr_current = get_lr(optimizerG)
    # clip2 = 0.01/lr_current
    # nn.utils.clip_grad_norm_(netG.parameters(),clip2)
    # scheduler.step(valid_gloss)
final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
