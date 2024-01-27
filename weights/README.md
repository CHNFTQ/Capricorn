# Capricorn weights computation



1. **Random downsample the data with different seeds and transform them.**
* replace `CELLLINE` with the `CELLLINE` you want(e.g. GM12878 and K562)
* The following command are for downsampling ratio of 16 only. Please change the ratio according to your need.

```
#Run this for different {i}(e.g. i=0,1,2,3,4):

python -m data_processing.Downsample -c CELLLINE -lr 10kb_d16 -r 16 --seed {i} 
python -m data_processing.Transform -c CELLLINE -lr 10kb_d16_seed{i} --cutoff 100
```

1. **Compute channel variance to get the initial channel weights.**
* Calculate the mean variance of each channel. The variances will be a initial estimate of channel difficulties.
* Replace `CELLLINE` and `YOUR_SEEDS` based on your need.
* The output results have to be manually saved to a .json file. See `weights/Multi_channel_weights.json` as an example.
* We have calculated initial channel weights for different experiments under the `weights` directory:
    * `no_weights.json` : Do not apply any weights.
    * `Multi_channel_weights.json` : Use the default transforms and use downsample rate 16.
    * `Multi_d32_channel_weights.json` : Use the default transforms and use downsample rate 32.
    * `TADagg_channel_weights.json` : Replace the TAD notation transform with aggregated TAD notation, and use downsample rate 16.
    * `Multi_crosschr_channel_weights.json` : Use the default transforms and use downsample rate 16. Run experiments on cross chromosome setting instead of cross cell-line setting.

```
python -m weights.Calculate_initial_weights  -c CELLLINE -lr 10kb_d16 -sd YOUR_SEEDS
```
2. **Compute difficulty to get the final channel weights.**
* Run Capricorn on the initial weights, then use the losses of each channel on validation set to adjust channel weights. 
* Replace `weights_file` with the initial weight file.
* The output results have to be manually saved to a .json file. See `weights/Multi_channel_weights_with_difficulty_modifier.json` as an example.
* We have calculated channel weights for different experiments under the `weights` directory:
    * `no_weights.json` : Do not apply any weights.
    * `Multi_channel_weights_with_difficulty_modifier.json` : Use the default transforms and use downsample rate 16.
    * `Multi_d32_channel_weights_with_difficulty_modifier.json` : Use the default transforms and use downsample rate 32.
    * `TADagg_channel_weights_with_difficulty_modifier.json` : Replace the TAD notation transform with aggregated TAD notation, and use downsample rate 16.
    * `Multi_crosschr_channel_weights_with_difficulty_modifier.json` : Use the default transforms and use downsample rate 16. Run experiments on cross chromosome setting instead of cross cell-line setting.
```
python -m weights.Calculate_difficulty_modifier --cell-line GM12878
```
