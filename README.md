# Capricorn

The python implementation of Capricorn. 

Capricorn is a tool for HiC contact matrix enhancement. Capricorn combines small-scale chromatin features with HiC matrix and utilize the diffusion model to generate High-coverage HiC matrices.

For more information, please read the preprint paper in (https://www.biorxiv.org/content/10.1101/2023.10.25.564065v2). 

## Dependencies
The following packages are required. We also provided recommended versions.
- Python 3.9
- Pytorch 2.0.1
- Numpy 1.24.3
- Scipy 1.10.1
- Pandas 1.5.3
- Scikit-learn 1.2.2
- Matplotlib 3.7.1
- tqdm 4.65.0
- Imagen-pytorch 1.25.11
- cooler 0.9.3
- einops 0.7.0
- einops-exts 0.0.4
- hic-straw 1.3.1
- chromosight 1.6.3

## Data preprocessing

1. **Set the enrivonment variables**
* In `dataset_information.py`, set your root directory for data. For example, let the root directory be `/data/hic_data`.
* Also, set the directory name to store different data file.
    * RAW_dir: stores raw hic data
    * hic_matrix_dir: stores the hic matrices in npz format.
    * multichannel_matrix_dir: stores the transformed hic matrices with additional views
    * data_dir: stores the data for training, validation and test.


2. **Download raw HiC data.**

* Download the raw HiC data from [GSE62525
GEO accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525)(Rao *et al.* 2014). We used [GM12878](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Fprimary%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
primary intrachromosomal and [K562](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz)
intrachromasomal.
* Create the `RAW_dir` directory under your root directory. Unzip the raw 
HiC data into the directory. This would create a new directory with the cellline name under the directory.

2. **Run the all-in-one scrips to generate data for training, validation and testing**

* The scripts will run several scripts to get the data for training, validation and test:
    1. Read raw data files from RAW_dir, save them in numpy matrix style(.npz files) in `hic_matrix_dir`.
    2. Read the high-coverage numpy matrices and downsample them.
    3. Transform both high-coverage and low-coverage matrices to include additional views. 
    4. Split the transformed matrices to get 40 by 40 patches, which can then be used for training, validation and test.

* See the codes for more details.

```
python -m data_processing.Preprocess -c GM12878
python -m data_processing.Preprocess -c K562
```



## Run Capricorn

1. **Calculate channel weights**
* We have prepared the channel weights in `./weights`.
* For the explanation of each weight file and how to get them, see the README in `./weights`.

3. **Train the model**
* This will train our model with the default parameters on GM12878. The checkpoints and train details are saved under `./checkpoints/{start_time}_Capricorn_GM12878`. You could replace `GM12878` with the cellline you need.
```
python train_Capricorn.py --cell-line GM12878
```
4. **Enhance HiC matrices with the model**
* This will use our trained model to enhance the HiC matrices on another cell-line. The results are saved in the same directory as the checkpoint.
* Replace `checkpoint_name`, `data_file` and `CELLLINE` according to your need.
```
python infer_Capricorn.py -ckpt checkpoint_name -f data_file -c CELLLINE
```
## Evaluating results
1. **Test the metrics**
* We provide a all-in-one evaluation scripts. See `HiC_evaluation/batch_evaluation.py` for more details.
* In the scripts, there are evaluationg settings we used for the experiments in the Paper. You should modify the settings to meet your need. 
* The following command will call the `Example` experiment, which evaluating the low-coverage matrices(resolution `10kb_d16_seed0`) with loop F1 scores on GM12878 cell line.

```
python -m HiC_evaluation.batch_evaluation -e Example -pt
```
