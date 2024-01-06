# the Root directory for all raw and processed data
root_dir = '/data/hic_data'  # Example of root directory name

res_map = {'5kb': 5_000, '10kb': 10_000, '25kb': 25_000, '50kb': 50_000, '100kb': 100_000, '250kb': 250_000,
           '500kb': 500_000, '1mb': 1_000_000}

# 'train' and 'valid' can be changed for different train/valid set splitting
set_dict = {
    'test' : (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X'),
    
    'train': (1,2,3,    6,7,8,9,10,   12,13,   15,16,17,18,19,20,21,22,'X'),
    'valid': (      4,5,           11,      14                            ),
    
    'test_crosschromosome' :(      4,                      14,   16,         20),
    'train_crosschromosome':(1,  3,  5,  7,8,9,   11,   13,   15,   17,18,19,   21,22),
    'valid_crosschromosome':(  2,      6,      10,   12),
    }

abandon_chromosome_dict = {
    'GM12878' : [],
    'K562' : [9]
}
