import os
import numpy as np

from snap.nsd_data import get_neural_data

import argparse

parser = argparse.ArgumentParser(description='Input parameters for running SNAP analysis on models')
parser.add_argument('-M', '--MODELS', metavar='--M', type=str, default='resnet18', help='Comma-separated string of models to analyse')
parser.add_argument('-R', '--REGIONS', metavar='--R', type=str, default='Early Visual Cortex', help='Comma-separated string of brain regions to run the regression with')
parser.add_argument('-A', '--ACTIVATION_POOLING', metavar='--A', type=list, default=[None], help='List of activation pooling methods to use')
parser.add_argument('-P', '--RANDOM_PROJECTION_DIM', metavar='--P', type=int, default=None, help='Number of dimensions to project neural data to')
parser.add_argument('-REG', '--REGULARIZATION', metavar='--REG', type=float, default=None, help='Regularization parameter to use in regression')
parser.add_argument('-SK', '--SKLEARN', metavar='--SK', type=bool, default=False, help='If true, uses sklearn ridge regression')
parser.add_argument('-DD', '--DEEPDIVE', metavar='--DD', type=bool, default=False, help='If true, uses deepdive modified ridge regression')
parser.add_argument('-T', '--TRAINING', metavar='--T', type=bool, default=None, help='If None, uses both trained and untrained. If True, uses only trained.')
parser.add_argument('-N', '--NUM_SAMPLES', metavar='--N', type=int, default=None, help='If None, uses all available samples. Else, uses a random subset of samples.')
parser.add_argument('-PCA', '--PCA_FEATURES', metavar='--PCA', type=bool, default=False, help='If true, does PCA on features before evaluation. Else, uses random orthogonal projections.')
parser.add_argument('-APT', '--ALPHA_PER_TARGET', metavar='--APT', type=bool, default=False, help='If true, each voxel is fit separately during the regression.')
parser.add_argument('-E', '--EMPIRICAL_ONLY', metavar='--E', type=bool, default=False, help='If true, only does the empirical regression.')


parser.add_argument('-BS', '--BATCH_SIZE', metavar='--B', type=int, default=128, help='Batch size for the NSD dataloader')
parser.add_argument('-SH', '--SHUFFLE', metavar='--SH', type=bool, default=False, help='Whether to shuffle the NSD data order for the dataloader')
parser.add_argument('-W', '--WORKERS', metavar='--W', type=int, default=4, help='The number of CPUs being used')

parser.add_argument('-S', '--SAVE_LOC', metavar='--S', type=str, default='/mnt/ceph/users/alargen/small_nsd/snap_data/reg_0', help='Location to save the data to')
parser.add_argument('-D', '--DATA', metavar='--D', type=str, default='/mnt/ceph/users/alargen/small_nsd/DeepJuiceDev/juicyfruits/nsd_subset', help='Path to the preprocessed NSD data')

args=parser.parse_args()

modelNames = args.MODELS.split(',')
regionNames = args.REGIONS.split(',')
activation_pooling = args.ACTIVATION_POOLING
rand_proj_dim = args.RANDOM_PROJECTION_DIM
reg = args.REGULARIZATION
sk = args.SKLEARN
dd = args.DEEPDIVE
training = args.TRAINING
num_samples = args.NUM_SAMPLES
pca = args.PCA_FEATURES
alpha_per_target = args.ALPHA_PER_TARGET
empirical_only = args.EMPIRICAL_ONLY

if sk:
    from snap.regression_utils_sklearn import regression_metric
elif dd:
    from snap.regression_utils_dd import regression_metric
else:
    from snap.regression_utils import regression_metric

if training is None:
    training = [True, False]
elif training:
    training = [True]
else:
    training = [False]

if (reg is None) and not (sk or dd):
    reg = 1e-14

batch_size = args.BATCH_SIZE
shuffle = args.SHUFFLE
workers = args.WORKERS

data_root = args.SAVE_LOC
os.makedirs(data_root, exist_ok=True)
nsd_root = args.DATA

pretrained = {True: 'pretrained',
              False: 'untrained'
              }

loader_kwargs = {'batch_size': batch_size,
                 'shuffle': shuffle,
                 'num_workers': workers,
                 'pin_memory': True,
                }              

device = 'cuda'

# Loop through the analyses specified above.
for region in regionNames:
    # for model_name in modelNames:
    for test_subj in [1, 2, 5, 7]:
        print(f'\n\n\nHolding out subject {test_subj}')
        data_dir = os.path.join(data_root,
            f"human_baseline")
        data_fname = os.path.join(data_dir,
            f"{region}_test_subj{test_subj}.npz")
        os.makedirs(data_dir, exist_ok=True)
        print(f'Saving to {data_fname}')

        subj_subset = [1, 2, 5, 7]
        subj_subset.remove(test_subj)

        _, _, labels_train = get_neural_data(region=region,
                                            loader_kwargs=loader_kwargs,
                                            data_path=nsd_root, num_samples=num_samples, 
                                            subj_subset=subj_subset)

        _, _, labels_test = get_neural_data(region=region,
                                            loader_kwargs=loader_kwargs,
                                            data_path=nsd_root, num_samples=num_samples, 
                                            subj_subset=[test_subj])

        # Create the Experiment Class and pass additional metrics
        regression_kwargs = {'num_trials': 5,
                            'reg': reg,
                            'num_points': 5,
                            'with_pca': pca,
                            'alpha_per_target': alpha_per_target,
                            'empirical_only': True,
                            'name': f'subjects_{subj_subset[0]}_{subj_subset[1]}_{subj_subset[2]}'
                            }
        
        reg_results = regression_metric(labels_train, labels_test, None, **regression_kwargs)

        # Save all of the metrics so we can load them.
        np.savez(data_fname, reg_results=reg_results)
        
                
print('All done!')