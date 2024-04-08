#%%
from monai.data import SmartCacheDataset
import pandas as pd
import os
import torch

from monai.transforms import (Compose,
                              LoadImaged,
                              EnsureChannelFirstd,
                              Orientationd,
                              ScaleIntensityRanged,
                              ScaleIntensityRangePercentilesd,
                              CropForegroundd,
                              ResizeWithPadOrCropd,
                              Resized,
                              EnsureTyped
                              )
from monai.data import DataLoader
from src.transform.transform import ConvertToMultiChannelBasedOnBratsClassesd
import numpy as np    
    
class BraTS_Dataset(SmartCacheDataset):
    def __init__(self, df, img_size, transform=None, cache_num=1):
        super().__init__(df, transform=transform, cache_num=cache_num)
        self.df = df
        self.transform = transform
        self.basic_transform = Compose([
            # Add basic transforms
            LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            EnsureTyped(keys=["image", "label"], track_meta=False),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
            ScaleIntensityRangePercentilesd(keys=['image', 'label'], lower=3, upper=97, b_min=0, b_max=1, clip = True),
            # ScaleIntensityRanged(keys=['image', 'label'], a_min=0, a_max=100, b_min=0, b_max=1, clip = True),
            # ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=[256, 256, 160], method='symmetric'),
            Resized(keys=['image', 'label'], spatial_size=(128, 128, 128))
        ])
    def __getitem__(self, index):
        ## load numpy array from nifti data
        patient_id = self.df[index]['id']
        modalities = sorted(os.listdir(os.path.join(self.df[index]['path'])))
        modalities_img = [modalities[1], modalities[2], modalities[3], modalities[4]]
        modalities_lab = modalities[0]
        image = [os.path.join(self.df[index]['path'], i) for i in modalities_img]
        label = os.path.join(self.df[index]['path'], modalities[0])
        ret = {'image' : image, 'label' : label}
        ## apply basic transform
        ret = self.basic_transform(ret)
        ## apply transform
        if self.transform:
            ret = self.transform(ret)
        return ret['image'], ret['label'] #patient_id, 

    def __len__(self):
        return len(self.df)