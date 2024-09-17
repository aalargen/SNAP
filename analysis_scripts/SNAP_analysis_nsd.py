import os
import numpy as np

from snap.wrapper import TorchWrapper
from snap.experiment import Experiment
from snap.nsd_data import get_neural_data
import snap.models as models

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


parser.add_argument('-BS', '--BATCH_SIZE', metavar='--B', type=int, default=128, help='Batch size for the NSD dataloader')
parser.add_argument('-SH', '--SHUFFLE', metavar='--SH', type=bool, default=False, help='Whether to shuffle the NSD data order for the dataloader')
parser.add_argument('-W', '--WORKERS', metavar='--W', type=int, default=4, help='The number of CPUs being used')

parser.add_argument('-S', '--SAVE_LOC', metavar='--S', type=str, default='/mnt/ceph/users/alargen/small_nsd/snap_data/reg_0', help='Location to save the data to')
parser.add_argument('-D', '--DATA', metavar='--D', type=str, default='/mnt/ceph/users/alargen/small_nsd/', help='Path to the preprocessed NSD data')

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

### Run the analysis computing the theoretical and empirical generalization error
### and measures for error mode geometry

# Loop through the analyses specified above.
for region in regionNames:
    data_loader_neural, images, labels = get_neural_data(region=region,
                                            loader_kwargs=loader_kwargs,
                                            data_path=nsd_root, num_samples=num_samples)
    for model_name in modelNames:
        print(f'\n\n\nAnalyzing {model_name}')
        for pooling in activation_pooling:
            for trained in training:
                data_dir = os.path.join(data_root,
                    f"data_{pooling}_RandProj_{rand_proj_dim}")
                data_fname = os.path.join(data_dir,
                    f"{region}_data_{model_name}_{pretrained[trained]}.npz")
                os.makedirs(data_dir, exist_ok=True)
                print(f'Saving to {data_fname}')

                # Get the model
                model_kwargs = {'name': model_name,
                                'pretrained': trained,
                                'device': device}
                model, layers, identifier = models.get_model(**model_kwargs)
                model_wrapped = TorchWrapper(model,
                                             layers=layers,
                                             identifier=identifier,
                                             activation_pooling=pooling)

                # Create the Experiment Class and pass additional metrics
                regression_kwargs = {'num_trials': 5,
                                     'reg': reg,
                                     'num_points': 5,
                                     }

                metric_fns = [regression_metric]
                exp = Experiment(model_wrapped,
                                 metric_fns=metric_fns,
                                 rand_proj_dim=rand_proj_dim)

                # Extract the activations of the layers passed above
                # using data_loader (only uses the inputs)
                exp.get_activations(data_loader_neural)

                # Compute metrics
                metric_kwargs = {'debug': False,
                                 'epsilon': 1e-14
                                 } | regression_kwargs

                exp_metrics = exp.compute_metrics(images=images,
                                                  labels=labels,
                                                  **metric_kwargs)
                layers = exp_metrics['layers']

                # Save all of the metrics so we can load them.
                np.savez(data_fname, exp_metrics=exp_metrics,
                         layers=layers, metric_kwargs=metric_kwargs)
                
                del model, layers, model_wrapped, exp, exp_metrics
                
print('All done!')