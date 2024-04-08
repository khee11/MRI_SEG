# make the basic torch trainer on which I can modify and add more features.
# I will use this as a base for all the other models.
#%%
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset.Dataset import BraTS_Dataset
import wandb
from monai.config import print_config
from src import *
from src.trainer.trainer import Trainer
import warnings
from monai.utils.misc import set_determinism
import argparse

def str2bool(v):
    if v.lower() in ['true', '1', 'yes', 'y', 't', 'yep', 'yeah', 'yup', 'certainly', 'uh-huh']:
        return True
    else:
        return False

warnings.filterwarnings("ignore")
# cs = ConfigStore.instance() # 접근을 위한 객체 
# cs.store(name="hematoma_config", node=DictConfig)
def get_args():
    parser = argparse.ArgumentParser(description="Hematoma Expansion Model Training")
    #do_wandb?
    parser.add_argument('--do_wandb', type=str2bool, default=True, help='whether to use wandb or not')
    # Train Arguments
    parser.add_argument('--model', type=str, default = 'swinunetr', help='the name for model. e.g. resnet121')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='the name for optimizer. e.g. sgd, adam, adamw. defalut adamw')
    parser.add_argument('--initial_lr', type=float, default=1.e-2, help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers for dataloader')
    # Visualization
    parser.add_argument('--do_medcam', type=str2bool, default=False, help='whether to do gradcam or not')

    args = parser.parse_args()
    return args

def main(cfg):
    set_determinism(20195794)
    data = pd.read_csv('/data/HWKIM/unconverted/brats_df.csv')
    train_data, valid_data = train_test_split(data, test_size=0.2)
    train_dataset = BraTS_Dataset(df = train_data.to_dict('records'), img_size = config.input_size, cache_num=cfg.cache_num)
    valid_dataset = BraTS_Dataset(df = valid_data.to_dict('records'), img_size = config.input_size, cache_num=cfg.cache_num)
    print_config
    trainer = Trainer(config, train_dataset, valid_dataset)
    trainer.main()
    
if __name__ == "__main__":
    args = vars(get_args())
    config_dictionary = dict(
    params=args)
    run = wandb.init(project="BraTS", config=config_dictionary)
    config = run.config
    main(config)
#%%