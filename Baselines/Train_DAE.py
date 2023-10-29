# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCSR: https://github.com/PSI-Lab/HiCSR
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from Models.HiCARN_1_Loss import GeneratorLoss
from Models.HiCSR import DAE as Generator
from Arg_Parser import root_dir
import argparse
import random
import torch.nn as nn
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
cs = np.column_stack


def adjust_learning_rate(epoch):
    lr = 1e-4
    return lr
  
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--name', type=str, default='HiCARN_1_GM12878_total')
parser.add_argument('--train_name', type=str, default='GM12878_train')
parser.add_argument('--valid_name', type=str, default='GM12878_valid')
parser.add_argument('--save_epoch', type=int, default=99999)
parser.add_argument('--model', type=str, default='DAE')
parser.add_argument('--noise_scale', type=float, default=0.1)
parser.add_argument('--input_channels', type=int, default=1)
args = parser.parse_args()
args = args.__dict__
input_channels=args['input_channels']
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

num_epochs = 600
batch_size = 256

os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

# prepare training dataset
train_file = os.path.join(data_dir, f'Multi_{resos}_c{chunk}_s{stride}_ds{ds_train}_b{bound}_{pool}_{train_name}.npz')
train = np.load(train_file)

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

train_set = TensorDataset(train_data, train_target, train_inds)

# prepare valid dataset
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

netG = Generator(input_channels=input_channels).to(device)
# loss function
criterionG = nn.MSELoss()

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

mse_scores = []


best_vloss = 999999999
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
        if input_channels == 1:
          target = target[:,:1,:,:]
        step += 1
        batch_size = target.size(0)
        run_result['nsamples'] += batch_size

        real_img = target.to(device)
        z = target + args['noise_scale'] * torch.randn_like(target)
        z = z.to(device)
        z = torch.clamp(z, 0, 1)
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
                    'mse': 0, 'nsamples': 0}
    netG.eval()
    batch_mses = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            if input_channels == 1:
              val_hr = val_hr[:,:1,:,:]
            batch_size = val_hr.size(0)
            valid_result['nsamples'] += batch_size
            hr = val_hr.to(device)
            lr = hr + args['noise_scale'] * torch.randn_like(hr)
            lr = lr.to(device)
            lr = torch.clamp(lr, 0, 1)
            sr = netG(lr)

            sr_out = sr
            hr_out = hr
            g_loss = criterionG(sr, hr)

            valid_result['g_loss'] += g_loss.item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            valid_result['mse'] += batch_mse * batch_size
            valid_bar.set_description(
                desc=f"[Predicting in Test set] MSE: {valid_result['mse']:.4f}")
            batch_mses.append(batch_mse)
    mse_scores.append((sum(batch_mses) / len(batch_mses)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_vloss = valid_result['g_loss'] / valid_result['nsamples']
    if now_vloss < best_vloss:
        best_vloss = now_vloss
        print(f'Now, Best vloss is {best_vloss:.6f}')
        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))
    if epoch % args['save_epoch'] == 0:
        tmp_ckpt_file = f'{datestr}_epoch{epoch}g_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, tmp_ckpt_file))
final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
