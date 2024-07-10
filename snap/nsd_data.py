"""
NSD code adapted from tutorial found in DeepNSD repo here: https://github.com/ColinConwell/DeepNSD
Also, uses preprocessed NSD data like the demo found in the above repo
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import gdown, tarfile
from .brainscore_data import get_ordered_data

import os


#ROI names and groups as defined in DeepNSD demo metadata
ROI_names = ['V1v', 'V2v', 'V3v', 'V1d', 'V2d', 'V3d', 'hV4', 'OFA', 'FFA-1',
   'FFA-2', 'OWFA', 'VWFA-1', 'VWFA-2', 'EBA', 'FBA-1', 'FBA-2',
   'OPA', 'PPA', 'RSC']
ROI_groups = ['V1', 'V2', 'V3', 'V4', 'Face Processing', 'Word Processing',
   'Body Processing', 'Scene Processing']
ROI_levels = ['Early Visual Cortex', 'Mid to High Level Visual Cortex']


def extract_tar(tar_file, dest_path, delete = True):
    tar = tarfile.open(tar_file)
    tar.extractall(dest_path)
    tar.close()
    os.remove(tar_file)


def get_neural_data(region=None, loader_kwargs=None, image_transforms=None,
                    data_path=None, dataset='demo'):
   """
   Args:
      ROI (list of str),
         the ROI(s) to take voxels from
      loader_kwargs (dict|None)
         dictionary of kwargs to pass into the loader
         if None, passes in default kwargs
      image_transforms (torchvision.transforms|None)
         transforms to apply to the images before getting the image from the dataloader
         if None, uses default ImageNet model transforms
      data_path (str|None)
         location of preprocessed NSD data. If using demo data from DeepNSD, should be
         the path to the directory containing "natural_scenes_demo" directory. if you 
         don't have this data, then this can also be the path where you would like the
         demo data to be downloaded
         if None, uses the current directory
      dataset (str)
         name of dataset in data_path to use. dataset directory name should be in the
         form of "natural_scenes_{dataset}"
   
   Returns:
      data_loader_neural (torch.utils.data.DataLoader)
         pytorch dataloader of image/brain response pairs
      images (list of torch.Tensor)
         list of images from the data. order parallels labels['responses'] order
      response (dir)
         TODO: figure this part out
   """
   all_ROIs = ROI_names + ROI_groups + ROI_levels

   if region: assert region in all_ROIs, f'{region} is not a valid ROI'
   
   response_data, stimulus_data, _ = get_nsd(data_path, region, dataset)

   if loader_kwargs is None:
      loader_kwargs = {'batch_size': 128,
                        'shuffle': False,
                        'num_workers': 4,
                        'pin_memory': True,
                        }
      
   if image_transforms is None:
      transform = [transforms.Resize((224, 224), antialias=True),
                   transforms.ConvertImageDtype(torch.float32),
                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                  ]
      transform = transforms.Compose(transform)
   else:
      transform = image_transforms

   image_paths = stimulus_data.image_path
   responses = [response_data[col].to_numpy() for col in response_data.columns]
   ds = NSDImageDataset(image_paths, responses, transform)
   dataloader_neural = DataLoader(ds, **loader_kwargs)
   images, responses = get_ordered_data(dataloader_neural)
   print(f'Shape of images: {images.shape}\nShape of brain responses: {responses.shape}')
   
   response = {'responses': responses}

   return dataloader_neural, images, response
   


def get_nsd(data_path, region, dataset='demo'):
   """
   returns dataframes with NSD data inside
   Adapted from DeepNSD GitHub repo
   """
   if data_path is None:
      path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
   else:
      path_dir = data_path
   data_dir = os.path.join(path_dir, f'natural_scenes_{dataset}')
        
   response_path = os.path.join(data_dir, 'response_data.parquet')
   metadata_path = os.path.join(data_dir, 'metadata.csv')
   stimdata_path = os.path.join(data_dir, 'stimulus_data.csv')
   image_root = os.path.join(data_dir, 'stimulus_set')
        
   path_set = [response_path, metadata_path, stimdata_path]
   if not all([os.path.exists(path) for path in path_set]):
      print('Downloading small NSD data from Google Drive to {}'.format(data_dir))
      tar_file = '{}/natural_scenes_demo.tar.bz2'.format(path_dir)
      gdown.download('https://drive.google.com/uc?export=download&id=176Vygj8SMic_p9tZtgqw60GtSA50D_VG',
                     quiet = False, output = tar_file)
      extract_tar(tar_file, path_dir)
      
   stimulus_data = pd.read_csv(stimdata_path)
   
   response_data = pd.read_parquet(response_path).set_index('voxel_id')
   metadata = pd.read_csv(metadata_path).set_index('voxel_id')

   # rename ROI for saving purposes
   metadata['roi_level'] = metadata['roi_level'].replace('Mid / High Level Visual Cortex', 'Mid to High Level Visual Cortex')

   # more renaming ROIS
   metadata.loc[metadata['roi_name'].str.contains('V1'), 'roi_group'] = 'V1'
   metadata.loc[metadata['roi_name'].str.contains('V2'), 'roi_group'] = 'V2'
   metadata.loc[metadata['roi_name'].str.contains('V3'), 'roi_group'] = 'V3'
   metadata.loc[metadata['roi_name'].str.contains('V4'), 'roi_group'] = 'V4'

   # ROI selectivity 
   if region:
      if region in ROI_names:
         metadata = metadata[metadata['roi_name'] == region]
      elif region in ROI_groups:
         metadata = metadata[metadata['roi_group'] == region]
      else:
         metadata = metadata[metadata['roi_level'] == region]

      voxel_ids = metadata.index
      response_data = response_data.loc[voxel_ids]
      
   stimulus_data = stimulus_data.set_index('image_id').loc[response_data.columns].reset_index()
   stimulus_data['image_path'] = image_root + '/' + stimulus_data.image_name

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
      img = Image.open(self.image_paths[idx])
      img = transforms.functional.pil_to_tensor(img)
      response = self.responses[idx]

      if self.transform:
         img = self.transform(img)

      return img, response

