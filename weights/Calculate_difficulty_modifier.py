# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# The script to train Capricorn for calculating channel weights.
# --------------------------------------------------------

from imagen_pytorch import Unet, Imagen, ImagenTrainer
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging
from matplotlib import pyplot as plt 
import argparse
import json
import random
from dataset_informations import *

cs = np.column_stack

# whether using GPU for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='The seed to train the model')
# data param
parser.add_argument('--prefix', type=str, default= 'Multi', help='The prefix of the data files')
parser.add_argument('--resos', type=str, default= '10kb10kb_d16', help='The matrix resolutions(d for downsample rate, i.e. d16 means 16 times downsampled)')
parser.add_argument('--data-seed', type=int, default=0, help='The seed of data')
parser.add_argument('--chunk', type=int, default = 40, help='The size of submatrices')
parser.add_argument('--stride', type=int, default = 40, help='The stride to slice submatrices along x and y axis')
parser.add_argument('--train-diagonal-stride', type=int, default = 40, help='When training, allow the submatrix slicing window slides along the main diagonal direction with this stride')
parser.add_argument('--bound', type=int, default = 200, help='Only consider elements whose distance to the main diagonal is smaller to this boundary. Only keep submatrices with elements in this boundary.')
parser.add_argument('--cell-line', type=str, default = 'GM12878', help='The cell line to train on')
parser.add_argument('--train-dataset', type=str, default = 'train', choices = set_dict.keys())
parser.add_argument('--valid-dataset', type=str, default = 'valid', choices = set_dict.keys())
#train param
parser.add_argument('--num-epochs', type=int, default = 100)
parser.add_argument('--batch-size', type=int, default = 32)
parser.add_argument('--valid-per-step-num', type=int, default = 10000, help='Run validation for every this number of train steps')
parser.add_argument('--valid-sample-num', type=int, default = 1000000, help='The largest number of data generated in one validation run. When validation is very slow, consider reduce this number.')
parser.add_argument('--cosine-decay-max-steps', type=int, default = 500000, help='The max steps in cosine learning rate scheduler')
parser.add_argument('--method-name', type=str, default='difficulty_test')
parser.add_argument('--checkpoint-name', type=str, default = None)
parser.add_argument('--save-dir', type=str, default = './checkpoints', help='where to save the experiment')
parser.add_argument('--train-unet', type=int, default = 1, help='When using cascaded diffusion, control which unet is currently training.')
#channel param
parser.add_argument('--num-channels', type=int, default = 5, help='The total number of channels in the data file')
parser.add_argument('--input-channels', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='Control which of the channels are used as model input')
parser.add_argument('--output-channels', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='Control which of the channels are used as model output')
parser.add_argument('--weights-file', type=str, default = 'weights/Multi_channel_weights.json')
parser.add_argument('--reverse-channels', nargs='+', type=int, default=[3], help='Reverse some of channels to ensure all in the same form')
#model param
parser.add_argument('--base-chunk', type=int, default = None, help='When using cascaded diffusion, control the matrix size at the base unet.')
parser.add_argument('--unet-resnet-depth', type=int, default = 5, help='The depth of resnet in unet')
parser.add_argument('--unet-input-dim', type=int, default = 128, help='The number of dimensions at the first layer in unet')
parser.add_argument('--diffusion-steps', type=int, default = 5, help='When evaluate, run how many steps of diffusion.')

parser.add_argument('--final-difficulty-step', type=int, default = 100000, help='Take the mse of new tasks as the difficulty at this step')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


unet1 = Unet(
    cond_on_text = False,
    dim = args.unet_input_dim,
    channels = len(args.output_channels),
    cond_images_channels = len(args.input_channels),
    dim_mults = (1, 2, 4),
    num_resnet_blocks = args.unet_resnet_depth,
    layer_attns = (False, True, True),
    layer_cross_attns = False,
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

imagen = Imagen(
    condition_on_text = False,
    channels = len(args.output_channels),
    unets = (unet1, ),
    image_sizes = (args.chunk, ),
    timesteps = args.diffusion_steps,
).cuda()

trainer = ImagenTrainer(imagen, cosine_decay_max_steps=args.cosine_decay_max_steps).cuda()    

# data_dir: directory storing processed data
data_dir = os.path.join(root_dir, data_dir)

datestr = time.strftime('%m_%d_%H_%M')
visdom_str = time.strftime('%m%d')

# out_dir: directory storing checkpoint files
save_name = f'{datestr}_{args.method_name}_{args.cell_line}'
out_dir = os.path.join(args.save_dir, save_name)
os.makedirs(out_dir, exist_ok=True)

if args.checkpoint_name is not None:
    checkpoint = args.checkpoint_name
    trainer.load(checkpoint)

log_file_path = os.path.join(out_dir, "experiment.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    filename=log_file_path, 
    filemode='w', 
)

print(f'Number of parameters: {count_parameters(unet1)}')
logging.info(f'Number of parameters: {count_parameters(unet1)}')

with open(os.path.join(out_dir, "args.json"), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

with open(args.weights_file) as f:
    weights = json.load(f)

print('Channel weights: ', weights[args.cell_line])
logging.info(f'Channel weights: {weights[args.cell_line]}')

channel_weights = np.expand_dims(weights[args.cell_line], (0, 2, 3))

def data_normalize(data):
    if len(args.reverse_channels) > 0 :
        data[:, args.reverse_channels, :, :] = 1 - data[:, args.reverse_channels, :, :]
    data = data * channel_weights
    return data

# prepare training dataset
train_file = os.path.join(data_dir, f'{args.prefix}_{args.resos}_seed{args.data_seed}_c{args.chunk}_s{args.stride}_ds{args.train_diagonal_stride}_b{args.bound}_{args.cell_line}_{args.train_dataset}.npz')
train = np.load(train_file)

print('Train file: ', train_file)
logging.info(f'Train file: {train_file}')

train_data = torch.tensor(data_normalize(train['data'])[:, args.input_channels, :, :], dtype=torch.float)
train_target = torch.tensor(data_normalize(train['target'])[:, args.output_channels, :, :], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

train_set = TensorDataset(train_data, train_target, train_inds)

# prepare valid dataset
valid_file = os.path.join(data_dir, f'{args.prefix}_{args.resos}_seed{args.data_seed}_c{args.chunk}_s{args.stride}_ds{               args.stride}_b{args.bound}_{args.cell_line}_{args.valid_dataset}.npz')
valid = np.load(valid_file)

print('Valid file: ', valid_file)
logging.info(f'Valid file: {valid_file}')

valid_data = torch.tensor(data_normalize(valid['data'])[:, args.input_channels, :, :], dtype=torch.float)
valid_target = torch.tensor(data_normalize(valid['target'])[:, args.output_channels, :, :], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

train_loss = []
difficulties = [[] for c in range(args.num_channels)]

best_mse = np.ones(args.num_channels)
best_loss = 1
step = 0

train_result = {'nsamples': 0, 'loss': 0}
for epoch in range(1, args.num_epochs + 1):
    train_bar = tqdm(train_loader)
    for data, target, _ in train_bar:
        step += 1
        trainer.train()

        batch_size = data.size(0)
        train_result['nsamples'] += batch_size
        
        loss = trainer(target, cond_images = data, unet_number = args.train_unet)
        trainer.update(unet_number = args.train_unet)

        train_result['loss'] += loss * batch_size

        train_bar.set_description(
            desc=f"[{epoch}/{args.num_epochs}, {step}] Train Loss: {train_result['loss'] / train_result['nsamples']:.4f}")

        if step % args.valid_per_step_num == 0 or step == args.final_difficulty_step:
            valid_sample_result = {'mse': np.zeros(args.num_channels, dtype=float), 'nsamples': 0}

            with torch.no_grad():
                trainer.eval()
                valid_bar = tqdm(valid_loader)
                num = args.valid_sample_num
                for val_lr, val_hr, inds in valid_bar:
                    if num <= 0 :break
                    num -= 1

                    batch_size = val_lr.size(0)
                    valid_sample_result['nsamples'] += batch_size
                    lr = val_lr.to(device)
                    hr = val_hr.to(device)
                    sr = trainer.sample(cond_images=lr, batch_size=batch_size, use_tqdm=False)

                    for i, c in enumerate(args.output_channels):
                        batch_mse = ((sr[:, i, :, :] - hr[:, i, :, :]) ** 2).mean(dim=(-1,-2))
                        valid_sample_result['mse'][c] += batch_mse.sum()
                    
                    valid_bar.set_description(
                        desc=f"[Predicting in valid set] mse: {np.array2string(valid_sample_result['mse'] / valid_sample_result['nsamples'], precision=4, floatmode='fixed')}")

            for c in args.output_channels:
                difficulties[c].append(valid_sample_result['mse'][c] / valid_sample_result['nsamples'])
            
            logging.info(f"Step: {step}, \t Valid, \t mse: {['%0.4e' % difficulties[c][-1] for c in args.output_channels]}")

        if step == args.final_difficulty_step:
            difficulty = valid_sample_result['mse'] / valid_sample_result['nsamples']
            difficulty_modifier = np.sqrt(np.min(difficulty)/difficulty)
            difficulty_modifier[0] = 1 #do not apply for HiC channel
            weights_with_difficulty_modifier = np.array(weights[args.dataset]) * difficulty_modifier
            print('Weights with difficulty modifier: ', np.array2string(weights_with_difficulty_modifier, precision=8, separator=', ', floatmode='fixed'))
            logging.info('Weights with difficulty modifier: ' + np.array2string(weights_with_difficulty_modifier, precision=8, separator=', ',  floatmode='fixed'))
            exit(0)

