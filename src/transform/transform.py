from monai.transforms import (Compose,
                              LoadImaged,
                              EnsureChannelFirstd,
                              Orientationd,
                              ScaleIntensityRanged,
                              CropForegroundd,
                              ResizeWithPadOrCropd,
                              Resized,
                              MapTransform
                              )
import torch

def basic_transform(cfg):
    basic_transform = Compose([
        # Add basic transforms
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=['image'], axcodes='RAS'),
        CropForegroundd(keys=['image'], source_key='image'),
        ScaleIntensityRanged(keys=['image'], a_min=0, a_max=100, b_min=0, b_max=1),
        ResizeWithPadOrCropd(keys=['image'], spatial_size=[512, 512, 32], method='symmetric'),
        Resized(keys=['image'], spatial_size=(128, 128, 32))
        ])
    return basic_transform

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        d['label'] = torch.squeeze(d['label'])
        return d
