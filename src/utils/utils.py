from collections import namedtuple
from datetime import datetime
from enum import Enum
from sklearn.metrics import mean_squared_error, roc_auc_score

import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn

"""
Migrating from utils.py.bak
"""
def get_loss_func(task_type="classification"):
    if task_type == "classification":
        return nn.BCEWithLogitsLoss()
    elif task_type == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError("task_type must be classification or regression")
    
    
def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")


def get_criterion(criterion_name):
    """Return criterion by name.

    Args:
        criterion_name (str)
    """
    if criterion_name == "mse":
        return nn.MSELoss()
    elif criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif criterion_name == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            f"Criterion {criterion_name} has not been implemented."
        )
        

class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True



"""
Adding new tools
"""

def get_local_time():
    cur_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
    return cur_time


def create_dirs(dir_paths):
    if not isinstance(dir_paths, (list, tuple)):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def init_seed(seed=0, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    
def get_instance(module, name, config, **kwargs):
    """
    A reflection function to get model.
    Args:
        module ([type]): Package Name.
        name (str): Top level value in config dict. (backbone, classifier, etc.)
        config (dict): The parsed config dict.
    Returns:
        Corresponding instance.
    """
    if config[name]["kwargs"] is not None:
        kwargs.update(config[name]["kwargs"])

    if name == "model":
        features = kwargs.pop("features")
        abstract_params = ["embedding_dim", "criterions", "task_types"]
        for k in list(filter(lambda k: k in config, abstract_params)):
            kwargs[k] = config[k]
        named_tuple = namedtuple("named_tuple", kwargs.keys())
        kwargs = {
            "features": features, 
            "config": named_tuple(**kwargs),
        }

    return getattr(module, config[name]["name"])(**kwargs)
            
            
class SaveType(Enum):
    NORMAL = 0
    BEST = 1
    LAST = 2
    
    
# -*- TensorboardWriter to wrtie logs -*-
from torch.utils import tensorboard

class TensorboardWriter(object):
    def __init__(self, log_dir):
        self.step = 0
        self.writer = tensorboard.SummaryWriter(log_dir)
        self.tb_writer_funcs = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }

    def set_step(self, step):
        self.step = step

    def __getattr__(self, name):
        if name in self.tb_writer_funcs:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            raise RuntimeError

    def close(self,):
        self.writer.close()


from torch.optim.lr_scheduler import _LRScheduler
# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, config):
        # if self.multiplier < 1.:
        #     raise ValueError('multiplier should be greater thant or equal to 1.')
        self.optimizer = optimizer
        self.total_epoch = config["epoch"]
        self.warmup = config["warmup"]
        self.after_scheduler = self.get_after_scheduler(config)
        self.finish_warmup = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_after_scheduler(self, config):
        scheduler_name = config["lr_scheduler"]["name"]
        scheduler_dict = config["lr_scheduler"]["kwargs"]

        if self.warmup != 0:
            if scheduler_name == "CosineAnnealingLR":
                scheduler_dict["T_max"] -= self.warmup - 1
            elif scheduler_name == "MultiStepLR":
                scheduler_dict["milestones"] = [
                    step - self.warmup + 1 for step in scheduler_dict["milestones"]
                ]

        if scheduler_name == "LambdaLR":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=eval(config["lr_scheduler"]["kwargs"]["lr_lambda"]),
                last_epoch=-1,
            )

        return getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer=self.optimizer, **scheduler_dict
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup - 1:
            self.finish_warmup = True
            return self.after_scheduler.get_last_lr()

        return [
            base_lr * float(self.last_epoch + 1) / self.warmup
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
            if self.finish_warmup and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
