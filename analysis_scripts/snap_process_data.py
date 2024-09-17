import os
import numpy as np
from itertools import product

from snap.wrapper import TorchWrapper
from snap.experiment import Experiment
from snap.nsd_data import get_neural_data
from snap.regression_utils import regression_metric
import snap.models as models
from snap.data_utils import DataProcess

import argparse

parser = argparse.ArgumentParser(description='Input parameters for running SNAP analysis on models')
parser.add_argument('-M', '--MODELS', metavar='--M', type=str, default='resnet18', help='Comma-separated string of models to process data from')
parser.add_argument('-R', '--REGIONS', metavar='--R', type=str, default='Early Visual Cortex', help='Comma-separated string of brain regions to process regression data from')
parser.add_argument('-A', '--ACTIVATION_POOLING', metavar='--A', type=list, default=[None], help='List of activation pooling methods to use')
parser.add_argument('-P', '--RANDOM_PROJECTION_DIM', metavar='--P', type=int, default=None, help='Number of dimensions to project neural data to')
parser.add_argument('-REG', '--REGULARIZATION', metavar='--REG', type=float, default=1e-14, help='Regularization parameter to use in regression')
parser.add_argument('-T', '--TRAINING', metavar='--T', type=bool, default=None, help='If None, uses both trained and untrained. If True, uses only trained.')

parser.add_argument('-S', '--SAVE_LOC', metavar='--S', type=str, default='/mnt/ceph/users/alargen/small_nsd/snap_data/reg_0/', help='Location to save the data to')

args=parser.parse_args()

modelNames = args.MODELS.split(',')
regionNames = args.REGIONS.split(',')
activation_pooling = args.ACTIVATION_POOLING
rand_proj_dim = args.RANDOM_PROJECTION_DIM
reg =  args.REGULARIZATION
training = args.TRAINING

data_root = args.SAVE_LOC
os.makedirs(data_root, exist_ok=True)

pretrained = {True: 'pretrained',
              False: 'untrained'
              }
if training is None:
    pass
elif training:
    del pretrained[False]
else:
    del pretrained[True]
    
### Process the data for the individual files
processed_data_root = os.path.join(data_root, 'processed/')

os.makedirs(processed_data_root, exist_ok=True)

rand_projections = ['None'] if rand_proj_dim is None else [rand_proj_dim]
activation_pooling = ['None' if pool is None else pool[:7] for pool in activation_pooling]
pooling_list = []
for item in product(activation_pooling, rand_projections):
    pooling_list += ["_RandProj_".join(item)]
activation_pooling = pooling_list.copy()

Data = DataProcess(data_root,
                   activation_pooling,
                   regionNames,
                   modelNames,
                   pretrained)
dfs_all = Data.get_dataframe(load=False, save_all_data_pckl=False)

sort_coord = 'final_scores'
threshold = 0.99

region_list = Data.region_list
pooling_list = Data.pooling_list
model_list = Data.model_list

for trained in pretrained.keys():
  for region in region_list:
      for pooling in pooling_list:
          print(f'Saving {region}, {pooling}')
          processed_data_name = os.path.join(processed_data_root,
                                             f'{region}_{pooling}_{pretrained[trained]}.npz')

          all_data_kwargs = dict(sort_coord=sort_coord,
                                 trained=trained,
                                 region_list=[region],
                                 pooling_list=[pooling],
                                 model_list=model_list,
                                 eff_dim_cutoff=0,
                                 threshold=threshold,
                                 )
          all_reg_hist, all_processed_data = Data.get_all_data(**all_data_kwargs)
          all_reg_hist = all_reg_hist[region][pooling]
          all_processed_data = all_processed_data[region][pooling]

          np.savez(processed_data_name,
                   all_reg_hist=all_reg_hist,
                   all_processed_data=all_processed_data,
                   all_data_kwargs=all_data_kwargs)
          
          del all_data_kwargs, all_reg_hist, all_processed_data

print('All done!')