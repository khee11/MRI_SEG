def get_callbacks(cfg):
    callback_list = []
    if 'earlystopping' in list(cfg.callbacks):
        callback_list.append(EarlyStopping(cfg.earlystopping_patience,
                                           cfg.earlystopping_verbose,
                                           cfg.earlystopping_delta
                                           )
                            )
    return callback_list


class Callback:
    def __init__(self, cfg):
        self._callbacks = get_callbacks(cfg)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    @property
    def callbacks(self):
        return self._callbacks


class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        print('callback: EarlyStopping ON.')
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return

        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
    def on_train_begin(self, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass
    def on_batch_begin(self, batch, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        pass


import time
import torch

class ProfileCallback:
    def __init__(self, ):
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        # Record the start time of the epoch
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Calculate and print the duration of the epoch
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")

        # Example of adding additional profiling, like memory usage (for GPU)
        if torch.cuda.is_available():
            print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    def on_train_begin(self, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass
    def on_batch_begin(self, batch, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        pass



class CALLBACKTEMPLATE:
    def __init__(self, ):
        print('callback: CALLBACKTEMPLATE activated.')
    def on_epoch_begin(self, epoch, logs=None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        pass 
    def on_train_begin(self, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass
    def on_batch_begin(self, batch, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        pass
