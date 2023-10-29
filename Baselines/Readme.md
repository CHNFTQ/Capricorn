## 1. Training Procedure for Baselines

#### 1.1 Preparation

First set the root_dir in Arg_parser.py (example: "HiCARN/Datasets_NPZ")

Then store your training/valid/test data in root_dir/data

#### 1.2 Arguments

+ device: the GPU you use
+ name: set the checkpoint name (which will be the suffix of the checkpoint in root_dir/checkpoints)
+ train_name: the suffix for the training data
+ valid_name: the suffix for the validation data
+ save_epoch: the frequency for storing checkpoint
+ model: the type of model for training(do not change)
+ input_channels: the number of input channels. 1 for only Hi-C experiments; 5 for 3Dto2D and multi-channel experiments(only valid for _multi.py)
+ output_channels: the number of output channels. 5 for only Hi-C and 3Dto2D experiments; 5 for multi-channel experiment(only valid for _multi.py)
+ cell_line(only for multi.py): the cell-line for training(GM12878/K562), deciding the weights for the multi-channel experiments(only valid for _multi.py)

#### 1.3 Detail for Baselines

Train HiCARN-1

+ python HiCARN_1_Train.py
+ python HiCARN_1_Train_multi.py --input_channels 5 --output_channels 5 (or 1) for multi-channel and 3Dto2D training

Train HiCARN-2

+ python HiCARN_2_Train.py
+ python HiCARN_2_Train_multi.py --input_channels 5 --output_channels 5 (or 1) for multi-channel and 3Dto2D training

Train HiCNN

+ python HiCNN_Train.py
+ python HiCNN_Train_multi.py --input_channels 5 --output_channels 5 (or 1) for multi-channel and 3Dto2D training

Train HiCSR:

+ python Train_DAE.py for training DAE(multichannel: python Train_DAE.py --input_channels 5)
+ python Train_HiCSR.py --DAE_dir YOUR_DAE_DIRECTORY --input_channels 5 (or 1) --output_channels 5 (or 1) for training HiCSR(setting input channels and output channels for multi-channel and 3Dto2D training)

The models will be stored in root_dir/checkpoints

## 2. Prediction

Generating data:

+ single channel:
  + HiCARN-1/2
    + python 40x40_Predict_With_Metrics.py -m HiCARN_1 -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_nonpool_GM12878_test.npz -c GM12878_HiCARN_1
  + HiCNN/HiCSR
    + python 28x28_Predict_With_Metrics.py -m HiCNN -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_nonpool_GM12878_test.npz -c GM12878_HiCNN
+ Multi-channel and 3Dto2D
  + HiCARN-1/2
    + python 40x40_Predict_With_Metrics_multi.py -m HiCARN_1_multi -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_nonpool_GM12878_test.npz -c GM12878_HiCARN_1 --input_channels 5 --output_channels 5(or 1)
  + HiCNN/HiCSR
    + python 28x28_Predict_With_Metrics_multi.py -m HiCNN_multi -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_nonpool_GM12878_test.npz -c GM12878_HiCNN --input_channels 5 --output_channels 5(or 1)

The predict matrices will be stored in root_dir/predict

## 3. Miscellaneous

+ Do not use downsample.py, Read_Data.py and generate.py in this folder to generate data. They are only directly cloned from [OluwadareLab/HiCARN: HiCARN: Resolution Enhancement of Hi-C Data Using Cascading Residual Networks (github.com)](https://github.com/OluwadareLab/HiCARN).
