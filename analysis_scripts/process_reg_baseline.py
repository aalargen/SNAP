import os
import numpy as np

import pickle

import argparse

parser = argparse.ArgumentParser(description='Input parameters for running SNAP analysis on models')
parser.add_argument('-R', '--REGIONS', metavar='--R', type=str, default='Early Visual Cortex', help='Comma-separated string of brain regions to run the regression with')
parser.add_argument('-S', '--SAVE_LOC', metavar='--S', type=str, default='/mnt/ceph/users/alargen/small_nsd/snap_data/pca_reg/human_baseline', help='Location to save the data to')

args=parser.parse_args()

regions = args.REGIONS.split(',')
base_path = args.SAVE_LOC
os.makedirs(base_path, exist_ok=True)

subjs = [1, 2, 5, 7]

err_keys = ['gen_errs', 'tr_errs', 'test_errs']
corr_keys = ['pearson_tr', 'pearson_test', 'pearson_gen']

all_data = {}
for region in regions:
    region_data = {}
    all_data[region] = region_data
    for subj in subjs:
        subj_data = np.load(f'{base_path}/{region}_test_subj{subj}.npz', allow_pickle=True)
        subj_data = subj_data['reg_results'].tolist()['cent']['responses']['responses']
        subj_dict = {}
        for key in err_keys:
            subj_dict[key] = subj_data[key].sum(axis=-1).mean(axis=0)
        for key in corr_keys:
            subj_dict[key] = subj_data[key].mean(axis=-1).mean(axis=0)
        region_data[subj] = subj_dict

avg_data = {}
all_keys = err_keys + corr_keys
for region, subj_dict in all_data.items():
    region_dict = {}
    for key in all_keys:
        total = np.zeros((2,))
        for subj in subjs:
            curr_data = subj_dict[subj][key]
            total += curr_data
        avg_val = total / len(subjs)
        region_dict[key] = avg_val
    avg_data[region] = region_dict

with open(f'{base_path}/avg_data.pickle', 'wb') as file:
    pickle.dump(avg_data, file, protocol=pickle.HIGHEST_PROTOCOL)
                
print('All done!')