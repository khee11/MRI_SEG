# make the basic torch trainer on which I can modify and add more features.
# I will use this as a base for all the other models.
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset.Dataset import Hematoma_Dataset
import wandb
import monai
from monai.config import print_config
from tqdm import tqdm
import os
from torch_lr_finder import LRFinder
from monai.utils import first
from src.utils import * 
from monai.utils.misc import set_determinism

def lr_finder(train_dataset, batch_size, num_epochs, learning_rate, callbacks): #valid_dataset, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = fetch_model('densenet121').to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=.01, num_iter=100)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state


def main():
    set_determinism(20195794)
    data = pd.read_csv('/mnt/hdd/smchou/hematoma/data/preprocessed/nifti/final_ct/EHR_final_ct.csv')
    train_data, valid_data = train_test_split(data, test_size=0.2)
    train_dataset = Hematoma_Dataset(train_data.to_dict('records'), (512, 512, 32), cache_num=10)
    # valid_dataset = Hematoma_Dataset(valid_data.to_dict('records'), cache_num=256)
    lr_finder(train_dataset, batch_size=8, num_epochs=1000, learning_rate=1.70e-7, callbacks=None)#valid_dataset,
    
         
if __name__ == "__main__":
    main()
#%%