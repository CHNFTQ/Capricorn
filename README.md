# Capricorn

The python implementation of Capricorn. Based on HiCARN (https://github.com/OluwadareLab/HiCARN) and imagen-pytorch (https://github.com/lucidrains/imagen-pytorch).

Following the instructions to run the experiment that use GM12878 cell-line to train and valid, use K562 cell-line to test with the resolution of 10kb and the downsample rate of 16.

## Dependencies
The following versions are recommended:
- Python 3.9
- Pytorch 2.0.1
- Numpy 1.24.3
- Scipy 1.10.1
- Pandas 1.5.3
- Scikit-learn 1.2.2
- Matplotlib 3.7.1
- tqdm 4.65.0
- Imagen-pytorch 1.25.11

## Data preprocessing

1. **Download raw HiC data.**

* Download the raw HiC data from [GSE62525
GEO accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525)(Rao *et al.* 2014). We used [GM12878](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Fprimary%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
primary intrachromosomal and [K562](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
intrachromasomal.
* In `HiCARN/Arg_Parser.py`, set your root directory for data. For example, let the root directory be `./hic_data`
* Create a `raw` directory under your root directory. Unzip the raw 
HiC data into the `raw` directory. This would create a new directory with the cellline name under the directory.

2. **Read the raw data(with HiCARN implementation).**

   * This will create a new directory `./hic_data/mat/GM12878` where all chrN_10kb.npz files will be stored.

```
cd ./HiCARN
python Read_Data.py -c GM12878
cd ../
```

3. **Downsample the data with specific seed.**

   * This will downsample original data with the downsample rate 16 and the seed 0, and then store all downsampled data files chrN_10kb_d16.npz into `./hic_data/mat/GM12878`.

```
python -m data_processing.Downsample -hr 10kb -lr 10kb_d16 -r 16 -c GM12878 --seed 0
```

4. **Transform the data to 3d matrices.**
   * This will transform original data and downsampled data with the specified transforms(Default: Normalized HiC, O/E normalize, TAD notation, Loop p-values, Loop ratios), and then store all 3d matrices files chrN_10kb_d16.npz into `./hic_data/multichannel_mat/$TRANSFORM_NAMES/GM12878`.
```
python -m data_processing.Transform -hr 10kb -lr 10kb_d16_seed0 -lrc 100 -chunk 40 -stride 40 -bound 200  -scale 1 -c GM12878 -s test
```

6. **Generate 40x40 submatrices for training, validation and testing**
   * Split the 3d matrices into 40 by 40 patches and generate Train, test and valid file. 
   * The data file will be stored under `./hic_data/data` with file name `Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_nonpool_GM12878_{train/valid/test}.npz`
```
python -m data_processing.Generate -hr 10kb -lr 10kb_d16_seed0 -lrc 100 -chunk 40 -stride 40 -bound 200  -scale 1 -c GM12878 --diagonal-stride 40 --save-prefix Multi -s train 

python -m data_processing.Generate -hr 10kb -lr 10kb_d16_seed0 -lrc 100 -chunk 40 -stride 40 -bound 200  -scale 1 -c GM12878  --diagonal-stride 40 --save-prefix Multi -s valid

python -m data_processing.Generate -hr 10kb -lr 10kb_d16_seed0 -lrc 100 -chunk 40 -stride 40 -bound 200  -scale 1 -c GM12878  --diagonal-stride 40 --save-prefix Multi -s test
```

5. **Preparing the target matrices which have the same normalization over HiC matrices.**
   * For convenience, transform the original matrix to Normalized HiC for evaluation. 
```
python -m data_processing.Transform -hr 10kb -lr 10kb_d16_seed0 -lrc 100 -chunk 40 -stride 40 -bound 200  -scale 1 -c GM12878 -s test --transform-names HiC
```

## Run Capricorn

1. **Compute channel variance to get the initial channel weights.**
    * This will random downsample multiple times, and calculate the mean variance of each channel. The variances will be a initial estimate of channel difficulties.
    * The output results have to be manually saved to a .json file. See `weights/Multi_channel_weights.json` as an example.
```
python -m data_processing.Get_Variance -hr 10kb -lr 10kb_d16 -r 16 -c GM12878 -lrc 100 -bound 200 --seed-num 10 --old-root-dir root_dir --save-dir temporary_save_dir
```
2. **Compute difficulty to get the final channel weights.**
    * This will run an initial experiment and then use the mean square error of each channel on validation set as mew channel difficulties. The new difficulties are then used to further adjust channel weights.
    * The output results have to be manually saved to a .json file. See `weights/Multi_channel_weights_with_difficulty_modifier.json` as an example.
    * We have calculated a series of channel weights for different experiments under the `weights` directory:
        * `no_weights.json` : Do not apply any weights.
        * `Multi_channel_weights_with_difficulty_modifier.json` : Use the default transforms and use downsample rate 16.
        * `Multi_d32_channel_weights_with_difficulty_modifier.json` : Use the default transforms and use downsample rate 32.
        * `TADagg_channel_weights_with_difficulty_modifier.json` : Replace the TAD notation transform with aggregated TAD notation, and use downsample rate 16.
        * `Multi_crosschr_channel_weights_with_difficulty_modifier.json` : Use the default transforms and use downsample rate 16. Run experiments on cross chromosome setting instead of cross cell-line setting.       
```
CUDA_VISIBLE_DEVICES=0 python compute_difficulty.py --dataset GM12878 --weights-file weights_file
```

3. **Train the model(with default parameters)**
    * This will train our model. The checkpoints and train details are saved under `./checkpoints/{start_time}_Capricorn_GM12878`.
```
CUDA_VISIBLE_DEVICES=0 python train_diffusion_40.py --dataset GM12878 --method-name Capricorn --weights-file weights_file
```
4. **Predict with the model(with default parameters)**
    * This will use our trained model to predict on another cell-line. The predicted results are saved in the same directory of the checkpoint.
    * You need to run data preprocessing for K562 cell-line in advance.
```
CUDA_VISIBLE_DEVICES=0 python eval_diffusion_40.py -m diffusion -lr 10kb_d16_seed0 -ckpt checkpoint_name -f data_file -c K562
```
## Evaluating results
1. **Test the metrics**
    * This will evaluate predicted results with image-based metrics. 
    * We provide several metrics with the same usage:
        * `HiC_evaluation.Image_Metrics`: Image-based metrics(MSE, PSNR, SSIM)
        * `HiC_evaluation.HiCCUPs`: Loop F1 scores.
        * `HiC_evaluation.insulation_score`: TAD-related metrics(TAD boundary F1 scores, insulation score mse, insulation score difference L2 norm).
    * See codes for detailed usage.

```
python -m HiC_evaluation.Image_Metrics --predict-dir predict_save_dir --target-dir target_save_dir --bounding 200 --dataset GM12878
```
