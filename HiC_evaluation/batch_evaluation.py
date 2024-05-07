# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# The all-in-one script to evaluate Capricorn and baselines. 
# --------------------------------------------------------

import os
import argparse
from dataset_informations import *

# The setting used in our evaluation. You could add your personalized setting for the experiments.
# We removed some duplicated evaluation(e.g. Capricorn evaluation was repeated in many experiments, and we didn't rerun it many times.)
# Currently not include the submatrix comparison experiments. Please see submatrices_loop_calc.py for the experiments.
Evaluate_settings = {
    'Example_GM12878':
    {

        'convert_to_cool' : False,

        'run_HiCCUPS': False,
        'run_Chromosight': True,
        'run_Mustache': True,

        'compare_MSE': False,
        'compare_HiCCUPS': False,
        'compare_Chromosight': True,
        'compare_Mustache': True,
        
        'dataset': 'test',
        'cell_line': 'GM12878',

        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb_d16_seed0', 
                'multiple': 1,
                'caption': 'hic', 
                'save_dir': 'results/example/LC/GM12878'
            },
        ]
    },
    'Example_K562':
    {

        'convert_to_cool' : False,

        'run_HiCCUPS': False,
        'run_Chromosight': True,
        'run_Mustache': True,

        'compare_MSE': False,
        'compare_HiCCUPS': False,
        'compare_Chromosight': True,
        'compare_Mustache': True,
        
        'dataset': 'test',
        'cell_line': 'K562',

        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb_d16_seed0', 
                'multiple': 1,
                'caption': 'hic', 
                'save_dir': 'results/example/LC/K562'
            },
        ]
    },
    'Main_GM12878':
    {
        'convert_to_cool' : True,

        'run_HiCCUPS': True,
        'run_Chromosight': True,
        'run_Mustache': True,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': True,
        'compare_Mustache': True,
        
        'dataset': 'test',
        'cell_line': 'GM12878',

        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': f'{root_dir}/mat/GM12878', 
                'res': '10kb_d16_seed0', 
                'multiple': 1,
                'caption': 'hic', 
                'save_dir': 'results/main_results/LC/GM12878'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/GM12878_HiCARN_1_K562_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCARN_1/GM12878'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/GM12878_HiCARN_2_K562_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCARN_2/GM12878'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/GM12878_HiCSR_K562_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCSR/GM12878'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/GM12878_HiCNN_K562_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCNN/GM12878'
            },
            {
                'data_dir': 'checkpoints/10_09_05_15_diffusion_3d_noaug_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic', 
                'save_dir': 'results/main_results/Capricorn/GM12878'
            },
        ]
    },
    'Main_K562': 
    {
        'convert_to_cool' : True,
        
        'run_HiCCUPS': True,
        'run_Chromosight': True,
        'run_Mustache': True,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': True,
        'compare_Mustache': True,
        
        'dataset': 'test',
        'cell_line': 'K562',
        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': f'{root_dir}/mat/K562', 
                'res': '10kb_d16_seed0', 
                'multiple': 1,
                'caption': 'hic', 
                'save_dir': 'results/main_results/LC/K562'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/K562_HiCARN_1_GM12878_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCARN_1/K562'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/K562_HiCARN_2_GM12878_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCARN_2/K562'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/K562_HiCSR_GM12878_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCSR/K562'
            },
            {
                'data_dir': '/data/liuyf/biocom/HiCARN/HiCARN/Datasets_NPZ/predict/K562_HiCNN_GM12878_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn', 
                'save_dir': 'results/main_results/HiCNN/K562'
            },
            {
                'data_dir': 'checkpoints/10_09_05_15_diffusion_3d_noaug_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic', 
                'save_dir': 'results/main_results/Capricorn/K562'
            },

        ]
    },
    'channel_ablation_GM12878': 
    {
        'convert_to_cool' : False,
        
        'run_HiCCUPS': True,
        'run_Chromosight': False,
        'run_Mustache': False,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': False,
        'compare_Mustache': False,
        
        'dataset': 'test',
        'cell_line': 'GM12878',

        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': 'checkpoints/10_06_15_58_diffusion_2d_noaugmentation_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H/GM12878'
            },
            {
                'data_dir': 'checkpoints/10_09_05_15_diffusion_3d_noaug_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F/GM12878'
            },
            {
                'data_dir': 'checkpoints/01_05_08_17_Capricorn_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F-L/GM12878'
            },
            {
                'data_dir': 'checkpoints/01_05_08_20_Capricorn_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F-T/GM12878'
            },
            {
                'data_dir': 'checkpoints/01_08_19_23_Capricorn_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H+O/GM12878'
            },
            {
                'data_dir': 'checkpoints/01_17_03_41_Capricorn_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F-O/GM12878'
            },
            {
                'data_dir': 'checkpoints/01_17_03_43_Capricorn_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H+L/GM12878'
            },
            {
                'data_dir': 'checkpoints/01_18_09_21_Capricorn_K562/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H+T/GM12878'
            }
        ]
    },
    'channel_ablation_K562': 
    {
        'convert_to_cool' : False,
        
        'run_HiCCUPS': True,
        'run_Chromosight': False,
        'run_Mustache': False,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': False,
        'compare_Mustache': False,
        
        'dataset': 'test',
        'cell_line': 'K562',
        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': 'checkpoints/10_06_08_42_diffusion_2d_noaugmentation_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H/K562'
            },
            {
                'data_dir': 'checkpoints/10_09_05_15_diffusion_3d_noaug_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F/K562'
            },
            {
                'data_dir': 'checkpoints/01_05_08_18_Capricorn_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F-L/K562'
            },
            {
                'data_dir': 'checkpoints/01_05_08_19_Capricorn_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F-T/K562'
            },
            {
                'data_dir': 'checkpoints/01_08_19_22_Capricorn_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H+O/K562'
            },
            {
                'data_dir': 'checkpoints/01_17_03_41_Capricorn_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/F-O/K562'
            },
            {
                'data_dir': 'checkpoints/01_17_03_42_Capricorn_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H+L/K562'
            },
            {
                'data_dir': 'checkpoints/01_18_09_20_Capricorn_GM12878/best_mse_0/predict/K562', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/channel_ablation_results/H+T/K562'
            }
        ]
    },
    'crosschromosome_GM12878':
    {
        'convert_to_cool' : False,
        
        'run_HiCCUPS': True,
        'run_Chromosight': False,
        'run_Mustache': False,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': False,
        'compare_Mustache': False,
        
        'dataset': 'test_crosschromosome',
        'cell_line': 'GM12878',

        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': f'{root_dir}/mat/GM12878', 
                'res' : '10kb_d16_seed0',
                'caption': 'hic',
                'save_dir': 'results/crosschromosome_results/LC/GM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_1_K562_base_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_1/GM12878_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_2_K562_base_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_2/GM12878_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCSR_K562_base_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCSR/GM12878_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCNN_K562_base_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCNN/GM12878_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_1_GM12878_base_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_1/GM12878_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_2_GM12878_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_2/GM12878_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCSR_GM12878_base_fixed', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCSR/GM12878_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCNN_GM12878_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCNN/GM12878_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/10_14_14_49_diffusion_3dcrosschr_noaug_K562/best_mse_0/predict/GM12878',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/crosschromosome_results/Capricorn/GM12878_TrainK562'
            },
            {
                'data_dir': 'checkpoints/10_14_14_49_diffusion_3dcrosschr_noaug_GM12878/best_mse_0/predict/GM12878', 
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/crosschromosome_results/Capricorn/GM12878_TrainGM12878'
            }, 
        ]
    },
    'crosschromosome_K562': 
    {
        'convert_to_cool' : False,
        
        'run_HiCCUPS': True,
        'run_Chromosight': False,
        'run_Mustache': False,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': False,
        'compare_Mustache': False,
        
        'dataset': 'test_crosschromosome',
        'cell_line': 'K562',
        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': f'{root_dir}/mat/K562',
                'res' : '10kb_d16_seed0',
                'caption': 'hic',
                'save_dir': 'results/crosschromosome_results/LC/K562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_1_K562_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_1/K562_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_2_K562_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_2/K562_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCSR_K562_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCSR/K562_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCNN_K562_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCNN/K562_TrainK562'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_1_GM12878_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_1/K562_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_2_GM12878_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCARN_2/K562_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCSR_GM12878_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCSR/K562_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCNN_GM12878_base_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/crosschromosome_results/HiCNN/K562_TrainGM12878'
            },
            {
                'data_dir': 'checkpoints/10_14_14_49_diffusion_3dcrosschr_noaug_K562/best_mse_0/predict/K562',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/crosschromosome_results/Capricorn/K562_TrainK562'
            },
            {
                'data_dir': 'checkpoints/10_14_14_49_diffusion_3dcrosschr_noaug_GM12878/best_mse_0/predict/K562',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/crosschromosome_results/Capricorn/K562_TrainGM12878'
            },
        ]
    },
    'IO_ablation_GM12878': 
    {
        'convert_to_cool' : False,
        
        'run_HiCCUPS': True,
        'run_Chromosight': False,
        'run_Mustache': False,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': False,
        'compare_Mustache': False,
        
        'dataset': 'test',
        'cell_line': 'GM12878',
        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/GM12878', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_1_K562_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_1/GM12878_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_1_K562_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_1/GM12878_I'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_2_K562_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_2/GM12878_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCARN_2_K562_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_2/GM12878_I'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCSR_K562_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCSR/GM12878_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCSR_K562_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCSR/GM12878_I'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCNN_K562_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCNN/GM12878_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/GM12878_HiCNN_K562_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCNN/GM12878_I'
            },
            {
                'data_dir': 'checkpoints/10_09_05_14_diffusion_3dto2d_noaug_K562/best_mse_0/predict/GM12878',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/IO_ablation_results/Capricorn/GM12878_I'
            },
        ]
    },
    'IO_ablation_K562': 
    {
        'convert_to_cool' : False,
        
        'run_HiCCUPS': True,
        'run_Chromosight': False,
        'run_Mustache': False,

        'compare_MSE': True,
        'compare_HiCCUPS': True,
        'compare_Chromosight': False,
        'compare_Mustache': False,
        
        'dataset': 'test',
        'cell_line': 'K562',
        'target_datas': {
            'MSE': {
                'data_dir' : f'{root_dir}/{multichannel_matrix_dir}/HiC/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'HiCCUPS': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Mustache': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
            'Chromosight': {
                'data_dir' : f'{root_dir}/{hic_matrix_dir}/K562', 
                'res': '10kb', 
                'multiple': 1,
                'caption': 'hic'
            },
        },
        'datas': [
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_1_GM12878_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_1/K562_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_1_GM12878_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_1/K562_I'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_2_GM12878_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_2/K562_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCARN_2_GM12878_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCARN_2/K562_I'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCSR_GM12878_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCSR/K562_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCSR_GM12878_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCSR/K562_I'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCNN_GM12878_multi_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCNN/K562_IO'
            },
            {
                'data_dir': 'checkpoints/baseline_ckpts/K562_HiCNN_GM12878_3to2_fixed',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hicarn',
                'save_dir': 'results/IO_ablation_results/HiCNN/K562_I'
            },
            {
                'data_dir': 'checkpoints/10_09_05_14_diffusion_3dto2d_noaug_GM12878/best_mse_0/predict/K562',
                'res': '10kb', 
                'multiple': 255,
                'caption': 'hic',
                'save_dir': 'results/IO_ablation_results/Capricorn/K562_I'
            },
        ]
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-pt', '--prepare-target-results', action='store_true')
    args = parser.parse_args()
    
    setting = Evaluate_settings[args.experiment]

    convert_to_cool = setting['convert_to_cool']

    run_HiCCUPS = setting['run_HiCCUPS']
    run_Chromosight = setting['run_Chromosight']
    run_Mustache = setting['run_Mustache']

    compare_MSE = setting['compare_MSE']
    compare_HiCCUPS = setting['compare_HiCCUPS']
    compare_Chromosight = setting['compare_Chromosight']
    compare_Mustache = setting['compare_Mustache']

    cell_line = setting['cell_line']
    dataset =  setting['dataset']

    target_datas = setting['target_datas']
    datas = setting['datas']

    if args.prepare_target_results:
        if run_HiCCUPS:
            data = target_datas['HiCCUPS']
            os.makedirs(f'{data["data_dir"]}/{data["res"]}', exist_ok=True)

            # Don't use external norms. If want to use some external norm, manaully do this convertion for data.
            command = f'python -m HiC_evaluation.HiCCUPS --data-dir {data["data_dir"]} --resolution {data["res"]} -c {cell_line} -s {dataset} --hic-caption {data["caption"]} --multiple {data["multiple"]} --external-norm-file NONE'
            os.system(command)
        
        if run_Chromosight:
            data = target_datas['Mustache']
            if convert_to_cool:
                os.makedirs(f'{data["data_dir"]}/{data["res"]}', exist_ok=True)

                # Don't use external norms. If want to use some external norm, manaully do this convertion for data.
                command = f'python -m data_processing.npz2cool --data-dir {data["data_dir"]} --resolution {data["res"]} -c {cell_line} -s {dataset} --hic-caption {data["caption"]} --multiple {data["multiple"]} --external-norm-file NONE'
                os.system(command)
            
            command = f'chromosight detect --pattern=loops_small --min-dist 15000 --max-dist 2000000 {data["data_dir"]}/bound200_res{data["res"]}.cool {data["data_dir"]}/{data["res"]}/chromosight_small'
            os.system(command)

        if run_Mustache:
            data = target_datas['Mustache']
            if convert_to_cool:
                os.makedirs(f'{data["data_dir"]}/{data["res"]}', exist_ok=True)

                # Don't use external norms. If want to use some external norm, manaully do this convertion for data.
                command = f'python -m data_processing.npz2cool --data-dir {data["data_dir"]} --resolution {data["res"]} -c {cell_line} -s {dataset} --hic-caption {data["caption"]} --multiple {data["multiple"]} --external-norm-file NONE'
                os.system(command)
            
            command = f'python -m HiC_evaluation.mustache -r 10kb -f {data["data_dir"]}/bound200_res{data["res"]}.cool -o {data["data_dir"]}/{data["res"]}/mustache.tsv'
            os.system(command)

    for data in datas:
        if data['save_dir'] is not None:
            os.makedirs(data['save_dir'], exist_ok=True)

        #some baselines have a different otuput name format. Change them to the standard format.
        command = f'rename 40kb 10kb {data["data_dir"]}/*'
        os.system(command)
        command = f'rename predict_c c {data["data_dir"]}/*'
        os.system(command)

        # prepare the output dir
        os.makedirs(f'{data["data_dir"]}/{data["res"]}', exist_ok=True)

        if compare_MSE:
            target_data = target_datas['MSE']
            # Never use multiple for MSE.
            command = f'python -m HiC_evaluation.Image_Metrics -t {target_data["data_dir"]} -tr {target_data["res"]} -p {data["data_dir"]} -pr {data["res"]} -pc {data["caption"]} -c {cell_line} -s {dataset}'
            if  data["save_dir"] is not None:
                command = command + f' -o {data["save_dir"]}/Image_metrics.tsv'
            os.system(command)

        if run_HiCCUPS:
            # Don't use external norms. If want to use some external norm, manaully do this convertion for data.
            command = f'python -m HiC_evaluation.HiCCUPS --data-dir {data["data_dir"]} --resolution {data["res"]} -c {cell_line} -s {dataset} --hic-caption {data["caption"]} --multiple {data["multiple"]} --external-norm-file NONE'
            os.system(command)

        if compare_HiCCUPS:
            target_data = target_datas['HiCCUPS']

            pred_file = f'{data["data_dir"]}/{data["res"]}/HiCCUPS/HiCCUPS_loop_annotation.bedpe'
            target_file = f'{target_data["data_dir"]}/{target_data["res"]}/HiCCUPS/HiCCUPS_loop_annotation.bedpe'
            command = f'python -m HiC_evaluation.bedpe_comparison -t {target_file} -p {pred_file} -d -s {dataset}'
            if  data["save_dir"] is not None:
                command = command + f' -o {data["save_dir"]}/HiCCUPS.tsv'
            os.system(command)
        
        if convert_to_cool:
            # Don't use external norms. If want to use some external norm, manaully do this convertion for data.
            command = f'python -m data_processing.npz2cool --data-dir {data["data_dir"]} --resolution {data["res"]} -c {cell_line} -s {dataset} --hic-caption {data["caption"]} --multiple {data["multiple"]} --external-norm-file NONE'
            os.system(command)
        
        if run_Chromosight:
            command = f'chromosight detect --pattern=loops_small --min-dist 15000 --max-dist 2000000 {data["data_dir"]}/bound200_{data["res"]}.cool {data["data_dir"]}/{data["res"]}/chromosight_small'
            os.system(command)

        if compare_Chromosight:
            target_data = target_datas['Chromosight']

            pred_file = f'{data["data_dir"]}/{data["res"]}/chromosight_small.tsv'
            target_file = f'{target_data["data_dir"]}/{target_data["res"]}/chromosight_small.tsv'
            command = f'python -m HiC_evaluation.bedpe_comparison -t {target_file} -p {pred_file} -d -s {dataset}'
            if  data["save_dir"] is not None:
                command = command + f' -o {data["save_dir"]}/chromosight.tsv'
            os.system(command)

        if run_Mustache:
            command = f'python -m HiC_evaluation.mustache -r 10kb -f {data["data_dir"]}/bound200_{data["res"]}.cool -o {data["data_dir"]}/{data["res"]}/mustache.tsv'
            os.system(command)

        if compare_Mustache:
            target_data = target_datas['Mustache']

            pred_file = f'{data["data_dir"]}/{data["res"]}/mustache.tsv'
            target_file = f'{target_data["data_dir"]}/{target_data["res"]}/mustache.tsv'
            command = f'python -m HiC_evaluation.bedpe_comparison -t {target_file} -p {pred_file} -d -s {dataset}'
            if  data["save_dir"] is not None:
                command = command + f' -o {data["save_dir"]}/mustache.tsv'
            os.system(command)
