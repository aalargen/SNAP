"""
NSD code adapted from DeepJuiceDev repo by Colin Conwell
Also, uses preprocessed NSD data provided in repo
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import gdown, tarfile
from .brainscore_data import get_ordered_data

import os, shutil


#ROI names and groups
functional_rois = ['V1v','V1d','V2v','V2d','V3v','V3d','hV4',
                  'FFA-1','FFA-2','OFA','EBA','FBA-1','FBA-2',
                        'OPA','PPA', 'VWFA-1','VWFA-2','OWFA']

midlevel_rois = {
    'V1': ['V1v','V1d'],
    'V2': ['V2v','V2d'],
    'V3': ['V3v','V3d'],
    'V4': ['hV4'],
    'Face Processing': ['FFA-1','FFA-2','OFA'],
    'Word Processing': ['VWFA-1','VWFA-2','OWFA'],
    'Body Processing': ['EBA','FBA-1','FBA-2'],
    'Scene Processing': ['OPA','PPA'], # no RSC?
}

global_rois = ['EVC','OTC']


### Helpers for loading in data

GDRIVE_HEADER = 'https://drive.google.com/uc?export=download&id='

def gdrive_download(download_id, down_path, dest_path=None, 
                    extract=True, delete_after=True, **kwargs):
    
    if not download_id.startswith('https://'):
        download_id = GDRIVE_HEADER + download_id
    
    gdown.download(download_id, down_path, **kwargs)
    
    # if download is tarball, extract it
    if 'tar' in down_path and extract:
        tarfile.open(down_path).extractall(dest_path)
    
    if delete_after: # delete the file after download
        if kwargs.get('quiet', False):
            print(f"Deleting download leftovers: {down_path}")
        
        shutil.rmtree(down_path) if os.path.isdir(down_path) else os.remove(down_path)


def load_pandas(path, root=None, **kwargs):
    if root is not None:
        path = os.path.join(root, path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file does not exist: {path}")

    # Get the file extension
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()
    
    excel_exts = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']

    # Match the file extension with the appropriate pandas read function
    if file_extension == '.csv':
        return pd.read_csv(path, **kwargs)
    if file_extension in excel_exts:
        return pd.read_excel(path, **kwargs)
    if file_extension == '.json':
        return pd.read_json(path, **kwargs)
    if file_extension == '.hdf':
        return pd.read_hdf(path, **kwargs)
    if file_extension == '.feather':
        return pd.read_feather(path, **kwargs)
    if file_extension == '.parquet':
        return pd.read_parquet(path, **kwargs)
    if file_extension == '.stata':
        return pd.read_stata(path, **kwargs)
    if file_extension == '.sas':
        return pd.read_sas(path, **kwargs)
    if file_extension == '.pkl':
        return pd.read_pickle(path, **kwargs)
    if file_extension == '.sql':
        # For .sql, a connection is required.
        raise NotImplementedError("Requires a database connection.")
    
    else: # attempt to load with read_table
        try: # takes any filename or buffer
            return pd.read_table(path, **kwargs)
        except Exception as error:
            raise ValueError(f"Failed to read pandas {file_extension}: {error}")


def load_data(data_path):
    if isinstance(data_path, str):
        return load_pandas(data_path).set_index('voxel_id')
    
    if isinstance(data_path, dict):
        response_data = []
        for key, value in data_path.items():
            response_data += [load_pandas(data_path[key])
                                .set_index('voxel_id')]
            
        return pd.concat(response_data)
    

def _parse_metadata_rois(all_rois, metadata, target_rois=None):
    if target_rois is None: # default
        target_rois = all_rois
    
    metadata.fillna(value={roi: 0 for roi in target_rois}, inplace=True)
    metadata[all_rois] = metadata[all_rois].astype(int)
    
    roi_voxel_counts = {roi: (metadata[roi] == True).sum() for roi in all_rois}
    roi_voxel_counts = dict(sorted(roi_voxel_counts.items(), 
                                        key=lambda x: x[1], reverse=True))
    return roi_voxel_counts


def get_roi_indices(metadata, all_rois, roi_subset=None, row_number=False):
    if 'voxel_id' in metadata.columns:
        metadata = metadata.set_index('voxel_id')

    if roi_subset is None:
        roi_subset = all_rois
    
    if not isinstance(roi_subset, list):
        roi_subset = [roi_subset]
        
    if row_number:
        metadata = metadata.reset_index()

    roi_indices = {}
    for roi in roi_subset:
        roi_subset = metadata[metadata[roi] == 1]
        
        roi_indices[roi] = {}
        for subj_id in roi_subset.subj_id.unique():
            subj_id_subset = roi_subset[roi_subset['subj_id'] == subj_id]
            roi_indices[roi][subj_id] = subj_id_subset.index.to_numpy()

    return roi_indices


### Main Data Loading Functions

def get_neural_data(region=None, loader_kwargs=None, image_transforms=None,
                    data_path=None, dataset='response', num_samples=None, 
                    num_voxels=None, shuffle_images=False, random_voxels=False,
                    subj_subset=[1, 2, 5, 7]):
   """
   Args:
      region (list of str),
         the ROI(s) to take voxels from
      loader_kwargs (dict|None)
         dictionary of kwargs to pass into the loader
         if None, passes in default kwargs
      image_transforms (torchvision.transforms|None)
         transforms to apply to the images before getting the image from the dataloader
         if None, uses default ImageNet model transforms
      data_path (str|None)
         location of preprocessed NSD data. Should lead to DeepJuiceDev/juicyfruits/nsd_subset
         if None, uses the current directory
      dataset (str)
         name of dataset in data_path to use. dataset directory name should be either
         'response' or 'demo_response'
      num_samples (int|None)
         number of images to use voxel data from
      num_voxels (int|None)
         number of voxels to get data from
      shuffle_images (boolean)
         shuffles the image data when true
      random_voxels (boolean):
         selects num_voxels randomly when true and num_voxels isn't None
      subj_subset (list of int),
         subjects to take data from
   
   Returns:
      data_loader_neural (torch.utils.data.DataLoader)
         pytorch dataloader of image/brain response pairs
      images (list of torch.Tensor)
         list of images from the data. order parallels labels['responses'] order
      response (dir)
         TODO: figure this part out
   """
   all_ROIs = functional_rois + list(midlevel_rois.keys()) + global_rois

   if region: assert region in all_ROIs, f'{region} is not a valid ROI'
   assert set(subj_subset).issubset({1, 2, 5, 7}), f'{subj_subset} contains one or more subjects not included in the data'
   
   response_data, stimulus_data, _ = get_nsd(data_path, region, subj_subset, dataset, 
                                             num_samples, num_voxels, random_voxels)

   if loader_kwargs is None:
      loader_kwargs = {'batch_size': 128,
                        'shuffle': False,
                        'num_workers': 4,
                        'pin_memory': True,
                        }
      
   if image_transforms is None:
      transform = [transforms.Resize(size=(224, 224), max_size=None, antialias=True),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                  ]
      transform = transforms.Compose(transform)
   else:
      transform = image_transforms

   image_paths = stimulus_data.image_path
   if shuffle_images:
      image_paths = image_paths.sample(frac=1, random_state=0).reset_index(drop=True)
   
   responses = [response_data[col].to_numpy() for col in response_data.columns]
   ds = NSDImageDataset(image_paths, responses, transform)
   dataloader_neural = DataLoader(ds, **loader_kwargs)
   images, responses = get_ordered_data(dataloader_neural)
   print(f'Shape of images: {images.shape}\nShape of brain responses: {responses.shape}')
   
   response = {'responses': responses}

   return dataloader_neural, images, response
   

def get_nsd(data_path, region=None, subj_subset=[1, 2, 5, 7], dataset='response', 
            num_samples=None, num_voxels=None, random_voxels=False,):
   """
   returns dataframes with NSD data inside
   Adapted from DeepNSD GitHub repo
   """
   if data_path is None:
      path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
   else:
      path_dir = data_path
   data_dir = os.path.join(path_dir, dataset)
   
   image_set = 'shared1000'

   stimulus_path = f'{path_dir}/stimulus/{image_set}.csv'
   image_root = os.path.join(path_dir, 'stimulus', image_set)

   response_path = {}
   metadata_path = {}
   path_set = [stimulus_path]
   for vset in ['EVC','OTC']:
      response_path[vset] = f'{path_dir}/response/{image_set}_{vset}/voxel_betas.csv'
      metadata_path[vset] = f'{path_dir}/response/{image_set}_{vset}/voxel_metas.csv'
      path_set += [response_path[vset], metadata_path[vset]]
      
   if not all([os.path.exists(path) for path in path_set]):
      print('Downloading response NSD data from Google Drive to {}'.format(data_dir))
      drive_id = '1R94PEyTfazeaD0M1YZx4-YJ2NYjQBoFQ'
      download_path = f'{path_dir}/response.tar.bz2'

      gdrive_download(drive_id, download_path, path_dir,
                     extract=True, delete_after=True)
      
   response_data = load_data(response_path)
   metadata = load_data(metadata_path)
   stimulus_data = load_pandas(stimulus_path)
   n_stimuli = len(stimulus_data)
   print(f'Number of images: {n_stimuli}')

   stimulus_data['image_path'] = image_root + '/' + stimulus_data.image_name

   all_rois = [roi for roi in metadata.columns if roi 
               in global_rois + functional_rois]

   metadata = metadata[['subj_id','ncsnr'] + all_rois]
   roi_voxel_counts = _parse_metadata_rois(all_rois, metadata)

   # adds midlevel ROIs to metadata
   if region in midlevel_rois:
      metadata[region] = metadata[midlevel_rois[region]].any(axis=1).astype(int)
      all_rois.append(region)

   if (num_samples is not None) and (num_samples < n_stimuli):
      stimulus_data = stimulus_data.sample(n=num_samples)
      samples = stimulus_data['image_id'].astype(str)
      response_data = response_data.loc[:, samples]

   # Reliability selection
   metadata = metadata[metadata['ncsnr'] > 0.2]

   # ROI + subject selection
   roi_indices = get_roi_indices(metadata, all_rois, roi_subset=region)
   
   all_idxs = []
   for roi, subj_dict in roi_indices.items():
      for subj in subj_subset:
         subj_idx = subj_dict[subj]
         all_idxs.extend(subj_idx)

   metadata = metadata.loc[all_idxs]
   metadata = metadata[~metadata.index.duplicated(keep='first')] # removes duplicates

   # this is mostly for sanity checks, so returns most reliable voxels instead of random sample
   if num_voxels is not None and num_voxels < metadata.shape[0]:
      if random_voxels:
         metadata = metadata.sample(n=num_voxels)
      else:
         metadata = metadata.nlargest(num_voxels, 'ncsnr')
      
   voxel_ids = metadata.index
   response_data = response_data.loc[voxel_ids]
   response_data = response_data[~response_data.index.duplicated(keep='first')] # removes duplicates
      
   stimulus_data = stimulus_data.set_index('image_id').loc[response_data.columns.astype(int)].reset_index()

   return response_data, stimulus_data, metadata


class NSDImageDataset(Dataset):
   """
   Dataset object for preprocessed NSD data

   Args:
      image_paths (list of str)
         list of paths to the corresponding images
      responses (list)
         list of numpy arrays or torch tensors representing the brain ressponse
      transform(torchvision.transforms)
         model-specific transforms to apply to the images
   """
   def __init__(self, image_paths, responses, transform):
      self.image_paths = image_paths
      self.responses = responses
      self.transform = transform

   def __len__(self):
      return len(self.responses)
   
   def __getitem__(self, idx):
      img = Image.open(self.image_paths[idx]).convert('RGB')
      response = self.responses[idx]

      if self.transform:
         img = self.transform(img)

      return img, response

