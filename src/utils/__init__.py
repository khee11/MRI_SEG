import monai.networks.nets as nets
import monai.networks.blocks as blocks 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#model
def fetch_model(model_name, spatial_dims = 3, in_channels = 4, out_channels = 3):
    model_name = model_name.lower()
    model_dict = {'swinunetr' : nets.SwinUNETR((128, 128, 128), spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels), 
                  }
    model = model_dict[model_name]
    return model


def fetch_optimizer(model_parameters, optimizer_name="adamw", lr=0.00001):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_parameters, lr=lr)
    elif optimizer_name == "adamw":
        return optim.AdamW(model_parameters, lr=lr)
    else:
        supported_optimizers = ["adam", "sgd", "rmsprop", "adamw"]
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers: {supported_optimizers}")





def fetch_scheduler(optimizer, scheduler_name="cosine_annealing_lr", **kwargs):
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "step_lr":
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "exponential_lr":
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == "cosine_annealing_lr":
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "reduce_lr_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "cyclic_lr":
        return lr_scheduler.CyclicLR(optimizer, **kwargs)
    else:
        supported_schedulers = ["step_lr", "exponential_lr", "cosine_annealing_lr", "reduce_lr_on_plateau", "cyclic_lr"]
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. Supported schedulers: {supported_schedulers}")


# metric
def update_metrics(metric_list, outputs, labels):
    for i, met in enumerate(metric_list):
        met.update(outputs, labels)
def reset_metrics(metric_list):
    for met in metric_list:
        met.reset()


def isnone(value):
    if (value == "None") or (value is None):
        return None
    else:
        return value

