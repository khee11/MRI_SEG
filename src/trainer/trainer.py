import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from src.utils import *
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR
from medcam import medcam
from torchsummary import summary
from src.callbacks import Callback, EarlyStopping
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


class Trainer:
    def __init__(self, cfg, train_dataset, valid_dataset):
        
        # Set the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_epochs = cfg.epochs
        self.callbacks = Callback(cfg)
        self.metrics = cfg.metrics
        self.do_medcam = cfg.params['do_medcam']
        # Create a model
        self.model = fetch_model(cfg.params['model'], spatial_dims=3, in_channels=4, out_channels=3).to(self.device)
        # for name, param in self.model.named_parameters():
        #     print(name, param.size())
        summary(self.model, (4, *cfg.input_size))
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
        if self.do_medcam:
            self.model = medcam.inject(self.model, output_dir=f"attention_maps", save_maps=True)        

        # Create a loss function
        self.criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")


        # Create an optimizer
        self.optimizer = fetch_optimizer(self.model.parameters(), optimizer_name=cfg.params['optimizer'], lr=cfg.params['initial_lr'])
        self.scheduler = fetch_scheduler(self.optimizer, scheduler_name="cosine_annealing_lr", T_max=self.num_epochs // 8, eta_min=0)
        # Create a data loader
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.params['batch_size'], shuffle=True, num_workers=cfg.params['num_workers'])
        if self.do_medcam:
            self.val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=cfg.params['num_workers'])
        else:
            self.val_loader = DataLoader(valid_dataset, batch_size=cfg.params['batch_size'] * 4, shuffle=False, num_workers=cfg.params['num_workers'])
        print(f'The number of train dataset: {len(train_dataset)}, The number of valid dataset: {len(valid_dataset)}')


    def train_one_epoch(self, epoch):
        self.callbacks.on_train_begin()
        # Train the model for one epoch
        self.model.train()
        if self.do_medcam:
            self.model.disable_medcam()
        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            self.callbacks.on_batch_begin(batch_index)
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # print(f'inputs: {inputs.size()}, labels: {labels.size()}')
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                if epoch == 0 and batch_index == 0:
                    print(f'\n inputs: {inputs.size()}, outputs: {outputs.size()}, labels: {labels.size()}')
                loss = self.criterion(outputs, labels)
                assert loss.dtype is torch.float32
            # Backward pass
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Print the loss for monitoring
            if batch_index == 1:
                wandb.log({'Loss': loss.item()})
        self.scheduler.step()
        self.callbacks.on_batch_end(batch_index)
        self.callbacks.on_train_end()
        torch.cuda.empty_cache()
        return loss.item()
    
    def valid_one_epoch(self, epoch):     
        metric_values = []
        metric_values_tc = []
        metric_values_wt = []
        metric_values_et = []               
        # Evaluate the model on validation set
        self.model.eval()
        if self.do_medcam:
            self.model.enable_medcam()

        with torch.no_grad():
            val_loss = 0
            # reset_metrics(self.metrics)
            for val_batch_index, (val_inputs, val_labels) in enumerate(self.val_loader):
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)
                val_outputs = self.model(val_inputs)
                if epoch == 0 and val_batch_index == 0:
                    print(f'val_inputs: {val_inputs.size()}, val_outputs: {val_outputs.size()}, val_labels: {val_labels.size()}')
                val_loss += self.criterion(val_outputs, val_labels).item()
                self.dice_metric(y_pred=val_outputs, y=val_labels)
                self.dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = self.dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = self.dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            self.dice_metric.reset()
            self.dice_metric_batch.reset()   
            val_loss /= len(self.val_loader)
            wandb.log({'epoch':epoch, 'Validation Loss': val_loss, 'metric_tc' : metric_tc, 'metric_wt': metric_wt, 'metric_et': metric_et})
        return val_loss

    def main(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.callbacks.on_epoch_begin(epoch)
            self.train_one_epoch(epoch)
            val_loss = self.valid_one_epoch(epoch)
            logs = {'epoch': epoch, 'val_loss': val_loss}
            self.callbacks.on_epoch_end(epoch, logs)
            if any(isinstance(cb, EarlyStopping) and cb.early_stop for cb in self.callbacks.callbacks):
                print("Stopping early due to no improvement.")
                break

        wandb.finish()
