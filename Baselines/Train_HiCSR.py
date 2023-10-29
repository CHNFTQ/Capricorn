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
from Models.HiCARN_2_Loss import GeneratorLoss
from Utils.SSIM import ssim
from math import log10
from torch import nn
from Models.HiCSR import DAE
from Arg_Parser import root_dir
import argparse
import random
cs = np.column_stack
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def adjust_learning_rate(epoch):
    lr = 1e-5
    return lr


# data_dir: directory storing processed data
# first run Train_DAE.py
data_dir = os.path.join(root_dir, 'data')
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default=root_dir)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--name', type=str, default='HiCSR_GM12878_total')
parser.add_argument('--train_name', type=str, default='GM12878_train')
parser.add_argument('--valid_name', type=str, default='GM12878_valid')
parser.add_argument('--DAE_dir', type=str)
parser.add_argument('--model', type=str, default='HiCSR')
parser.add_argument('--lambda_a', type=float, default=2.5e-3)
parser.add_argument('--lambda_1', type=float, default=1)
parser.add_argument('--lambda_f', type=float, default=1)
parser.add_argument('--input_channels', type=int, default=1)
parser.add_argument('--output_channels', type=int, default=1)
parser.add_argument('--cell_line', type=str, default="GM12878", choices=['GM12878', 'K562'])
args = parser.parse_args()
args = args.__dict__
input_channels=args['input_channels']
output_channels=args['output_channels']
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])
# out_dir: directory storing checkpoint files
out_dir = os.path.join(args['ckpt_dir'], 'checkpoints')
os.makedirs(out_dir, exist_ok=True)
if args['model'] == 'HiCSR':
    from Models.HiCSR import Generator, Discriminator
else:
    raise NotImplementedError
if args['cell_line'] == 'GM12878' and output_channels == 5:
    weight = [1, 0.072986629947397, 0.009831300898402,  0.15943821113032, 0.031014444455153]
elif args['cell_line'] == 'K562' and output_channels == 5:
    weight = [1, 0.035779129452585, 0.009994454264768, 0.136775029550876, 0.022922722050868]
else:
    weight = [1, 0, 0, 0, 0]
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
batch_size = 128
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])

# whether using GPU for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print("CUDA avalable?", torch.cuda.is_available())
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

if args['model'] == 'HiCSR':
    netG = Generator(input_channels=input_channels, out_channels=output_channels).to(device)
    netD = Discriminator(input_channels=output_channels).to(device)
else:
    raise NotImplementedError


class FeatureReconstructionLoss(nn.Module):
    def __init__(self, layers = [0,1,2,3,4]):
        super(FeatureReconstructionLoss, self).__init__()
        loss_net = DAE(input_channels=output_channels)
        loss_net.load_state_dict(torch.load(args['DAE_dir']))
        encoder = list(loss_net.children())[0]

        self.sub_nets = []
        self.layers = layers

        for layer in layers:
            list_of_layers = list(encoder)[:layer]
            final_layer = [encoder[layer][0]]
            sub_net = nn.Sequential(*(list_of_layers + final_layer)).float().eval().cuda()

            for param in sub_net.parameters():
                param.requires_grad = False

            self.sub_nets.append(sub_net)

        self.loss = nn.MSELoss()

    def forward(self, prediction, target):

        feature_loss = torch.tensor([0.0]).float().cuda()

        for net in self.sub_nets:
            pred_feat = net(prediction)
            target_feat = net(target)
            loss = self.loss(pred_feat, target_feat)
            feature_loss += loss

        return feature_loss

# loss function
adv_loss = nn.BCEWithLogitsLoss().to(device)
l1_loss = nn.L1Loss().to(device)
feature_reconstruction_loss = FeatureReconstructionLoss().to(device)
criterionG = GeneratorLoss().to(device)
criterionD = torch.nn.BCELoss().to(device)

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=1e-5)
optimizerD = optim.Adam(netD.parameters(), lr=1e-5)

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0
best_vloss = 999999
for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)
    optimizerD = optim.Adam(netD.parameters(), lr=alr)

    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()

    netG.train()
    netD.train()
    train_bar = tqdm(train_loader)
    step = 0
    for data, target, _ in train_bar:
        if input_channels == 1:
            data = data[:,:1,:,:]
        if output_channels == 1:
            target = target[:,:1,:,:]
        step += 1
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size
        real_img = target.to(device)
        
        if args['model'] == 'HiCSR':
            z = F.pad(data, (6, 6, 6, 6), mode='constant')
            z = z.to(device)
        else:
            z = data.to(device)
        fake_img = netG(z)
        netD.zero_grad()
        if output_channels > 1:
            image_loss = 0
            for i in range(output_channels):
                image_loss += l1_loss(fake_img[:,i,:,:], real_img[:,i,:,:]) * weight[i]
        else:
            image_loss = l1_loss(fake_img, real_img)
        real_out = netD(real_img)
        fake_out = netD(fake_img)
        d_loss_real = criterionD(real_out, torch.ones_like(real_out))
        d_loss_fake = criterionD(fake_out, torch.zeros_like(fake_out))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        netG.zero_grad()

        
        feature_loss = sum(feature_reconstruction_loss(fake_img, real_img))
        fake_out2 = netD(fake_img)
        GAN_loss = criterionD(fake_out2, torch.ones_like(fake_out2))
        g_loss = args['lambda_a'] * GAN_loss + \
                               args['lambda_1'] * image_loss + \
                               args['lambda_f'] * feature_loss
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size
        run_result['d_loss'] += d_loss.item() * batch_size
        run_result['d_score'] += real_out.mean().item() * batch_size
        run_result['g_score'] += fake_out.mean().item() * batch_size
        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] "
                 f"Loss_D: {run_result['d_loss'] / run_result['nsamples']:.4f} "
                 f"Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f} "
                 f"D(x): {run_result['d_score'] / run_result['nsamples']:.4f} "
                 f"D(G(z)): {run_result['g_score'] / run_result['nsamples']:.4f}")

    train_gloss = run_result['g_loss'] / run_result['nsamples']
    train_dloss = run_result['d_loss'] / run_result['nsamples']
    train_dscore = run_result['d_score'] / run_result['nsamples']
    train_gscore = run_result['g_score'] / run_result['nsamples']

    valid_result = {'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()
    netD.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            if input_channels == 1:
                val_lr = val_lr[:,:1,:,:]
            if output_channels == 1:
                val_hr = val_hr[:,:1,:,:]
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            if args['model'] == 'HiCSR':
                lr = F.pad(val_lr, (6, 6, 6, 6), mode='constant')
                lr = lr.to(device)
            else:
                lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = netD(sr)
            hr_out = netD(hr)
            d_loss_real = criterionD(hr_out, torch.ones_like(hr_out))
            d_loss_fake = criterionD(sr_out, torch.zeros_like(sr_out))
            d_loss = d_loss_real + d_loss_fake
            
            
            image_loss = l1_loss(hr, sr)
            feature_loss = sum(feature_reconstruction_loss(sr, hr))

            g_loss = image_loss
                                
                                
            valid_result['g_loss'] += g_loss.item() * batch_size
            valid_result['d_loss'] += d_loss.item() * batch_size
            valid_result['g_score'] += sr_out.mean().item() * batch_size
            valid_result['d_score'] += hr_out.mean().item() * batch_size

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
    valid_dloss = valid_result['d_loss'] / valid_result['nsamples']
    valid_gscore = valid_result['g_score'] / valid_result['nsamples']
    valid_dscore = valid_result['d_score'] / valid_result['nsamples']
    if valid_result['g_loss'] < best_vloss:
        best_vloss = valid_result['g_loss']
        print(f'Now, Best vloss is {best_vloss:.6f}')
        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))    

final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
final_ckpt_d = f'{datestr}_finald_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'

torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
torch.save(netD.state_dict(), os.path.join(out_dir, final_ckpt_d))
