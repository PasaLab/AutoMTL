import builtins
from collections import OrderedDict
import copy
import datetime
import json
import os
import sys
from time import time

import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger
from numpy.typing import NDArray
from typing import Any
import yaml

from src.models.nas import SuperNet
from src.datasets.dataset_utils import DataGenerator
from src.models.basic.features import DenseFeature, SparseFeature
from src.utils.utils import (
    get_loss_func, get_metric_func, get_instance, get_local_time, create_dirs, init_seed,
    GradualWarmupScheduler, SaveType, TensorboardWriter,
)


class ArchSearchRunManager:
    """A trainer for multi task learning.
    """
    def __init__(self, config, rank=0):
        self.rank = rank
        self.config = config
        self.config["rank"] = rank
        self.distribute = config["n_gpu"] > 1
        self.task_num = len(config["task_types"])
        self.loss_fns = [get_loss_func(task_type) for task_type in config["task_types"]]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in config["task_types"]]
        self.early_stop_patience = config["earlystop_patience"]
        self.early_stop_counter = 0
        self.val_per_epoch = config["val_per_epoch"]
        self.full_arch_train_epoch_idx = config["warmup_epochs"]
        self.fine_tune_epoch_idx = config["warmup_epochs"] + config["full_arch_train_epochs"]
        (
            self.result_path,
            self.log_path,
            self.ckpt_path,
            self.viz_path,              # tensorboard path
        ) = self._init_files(config)
        (
            self.device,
            self.list_ids,
        ) = self._init_device(rank, config)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.features,
        ) = self._init_dataloader(config)
        (
            self.net,
            self.best_model_weights,
        ) = self._init_model(config)
        (
            self.optimizer,
            self.arch_optimizer,
            self.scheduler,
            self.from_epoch,
            self.best_val_auc,
            # self.best_test_auc,
        ) = self._init_optim(config)
        self.logger = self._init_logger()
        print(config)


    def fit(self,):
        """ The normal train loop: train loop: train-val and save model val-cc increases
        """
        print(
            f"#alpha_params: {len(list(self.net.alpha_parameters()))}\t"
            f"#beta_params: {len(list(self.net.beta_parameters()))}\t"
            f"#weight_params: {len(list(self.net.weight_parameters()))}\t"
        )
        experiment_begin = time()
        for epoch_idx in range(self.from_epoch + 1, self.config["epoch"]):
            print("============ Train on the train set ============")
            print("{}, learning rate: {}".format(
                "Warm up" if epoch_idx < self.full_arch_train_epoch_idx
                else "{}".format(
                    "Full arch train" if epoch_idx < self.fine_tune_epoch_idx
                    else "Fine tune"
                ),
                self.scheduler.get_last_lr()
            ))
            self._train_one_epoch(epoch_idx)        # train one epoch
            
            # print current network architecture
            self.net.print_arch(epoch_idx)
            # discretize
            if epoch_idx >= self.full_arch_train_epoch_idx:
                for _ in range(self.config["discretize_ops"]):
                    self.net.discretize_one_op()
            if epoch_idx == self.fine_tune_epoch_idx:
                self.net.export_architecture()
                # net re-init
                # self.net.init_model()
                # self.optimizer = get_instance(
                #     torch.optim, "optimizer", self.config, params=self.net.parameters(),
                # )
                self.best_val_auc = np.zeros(self.task_num)
                self.best_model_weights = copy.deepcopy(self.net.state_dict())
            
            if ((epoch_idx + 1) % self.val_per_epoch) == 0:
                print("============ Validation on the val set ============")
                val_auc = self._validate(epoch_idx=epoch_idx, is_test=False)
                
                if epoch_idx >= self.full_arch_train_epoch_idx:     # early stop for fine tune
                    if self._compute_improvement(val_auc, self.best_val_auc) > 0:
                        self.best_val_auc = val_auc
                        self.early_stop_counter = 0
                        self.best_model_weights = copy.deepcopy(self.net.state_dict())
                        self._save_model(epoch_idx, SaveType.BEST)
                    elif self.early_stop_counter < self.early_stop_patience:
                        self.early_stop_counter += 1
                    else:                                                       # early stop
                        print(" * Early stopping, best_val_auc: {}".format(
                                " ".join([
                                    "Task#{}-({:.5f})".format(i, self.best_val_auc[i]) 
                                    for i in range(self.task_num)
                        ])))
                        self.net.load_state_dict(self.best_model_weights)
                        self._save_model(epoch_idx, SaveType.LAST)
                        break
                self._save_model(epoch_idx, SaveType.LAST)
                print(" * Best Auc: {}".format(
                        " ".join([
                            "Task#{}-({:.5f})".format(i, self.best_val_auc[i]) 
                            for i in range(self.task_num)
                        ])))

            time_scheduler = self._cal_time_scheduler(experiment_begin, epoch_idx)
            print(" * Time: {}".format(time_scheduler))
            self.scheduler.step()
            
        print(
            "End of experiment, took {}".format(
                str(datetime.timedelta(seconds=int(time() - experiment_begin)))
            ))
        print("Result DIR: {}".format(self.result_path))


    def evaluate(self, ) -> NDArray[Any]:
        """ The normal test loop: 
        Returns:
            numpy: Array of scores for tasks.
        """
        self.net.load_state_dict(self.best_model_weights)
        total_test_auc = np.zeros(self.task_num)
        
        print("============ Testing on the test set ============")
        for epoch_idx in range(self.config["test_epoch"]):
            test_auc = self._validate(epoch_idx, is_test=True)
            total_test_auc += test_auc
        avg_test_auc = total_test_auc / self.config["test_epoch"]
        print("* Aver Accuracy: {}".format(
            " ".join([
                "Task#{}-({:.5f})".format(
                    i, avg_test_auc[i]
                ) for i in range(self.task_num)
        ])))
        print("............Testing is end............")
        return avg_test_auc


    def _train_one_epoch(self, epoch_idx=0):
        """ The train stage.
        """
        self.net.train()
        batch_max_len = max(map(len, self.train_loader))
        
        total_loss = np.zeros(self.task_num)
        num_batches = len(self.train_loader)
        for batch_idx, (train_input, train_gt) in enumerate(self.train_loader):
            train_input = {k: v.to(self.device) for k, v in train_input.items()}  # tensor to GPU
            train_gt = train_gt.to(self.device)
            train_pred = self.net(train_input)
            train_loss = self._compute_loss(train_pred, train_gt)

            loss = sum(train_loss) / self.task_num
            self.net.zero_grad()
            loss.backward()
            if (epoch_idx >= self.full_arch_train_epoch_idx) and \
                (epoch_idx < self.fine_tune_epoch_idx):
                self.arch_optimizer.step()
            self.optimizer.step()

            total_loss += np.array([l.item() for l in train_loss])
            if ((batch_idx + 1) % self.config["log_interval"] == 0) or (
                (batch_idx + 1)  >= len(self.train_loader)
            ):
                print(
                    "Epoch-({}): [{}/{}]"
                    "\tLoss: {}".format(
                        epoch_idx,
                        (batch_idx + 1),
                        num_batches,
                        " ".join([
                            "Task#{}-({:.5f})".format(
                                i, total_loss[i] / (batch_idx + 1),
                            ) for i in range(self.task_num)
                        ]))
                )

            
    def _validate(self, epoch_idx=0, is_test=False):
        """ The val/test stage
        Returns:
            numpy: Array of scores for tasks.
        """
        self.net.eval()
        gts, preds = [], []
        dataloader = self.val_loader if not is_test else self.test_loader
        with torch.no_grad():
            for i, (test_input, test_gt) in enumerate(dataloader):
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_gt = test_gt.to(self.device)
                test_pred = self.net(test_input)
                gts.extend(test_gt.tolist())
                preds.extend(test_pred.tolist())
        scores = self._compute_score(preds=preds, gts=gts)
        print(
            "Epoch-({}): "
            " * Auc: {}".format(
                epoch_idx,
                " ".join([
                    "Task#{}-({:.5f})".format(
                        i, scores[i],
                    ) for i in range(self.task_num)
                ]))
        )
        return scores

    
    def _compute_loss(self, preds, gts):
        """
        Returns:
            tensor: Array of losses for tasks.
        """
        losses = torch.zeros(self.task_num).to(self.device)
        for i in range(self.task_num):
            losses[i] = self.loss_fns[i](preds[:, i], gts[:, i].float())
        return losses

    
    def _compute_score(self, preds, gts):
        """
        Return:
            numpy: Array of scores for tasks.
        """
        preds, gts = np.array(preds), np.array(gts)
        scores = np.zeros(self.task_num)
        for i in range(self.task_num):
            scores[i] = self.evaluate_fns[i](gts[:, i], preds[:, i])
        return scores
    
    
    def _compute_improvement(self, new_auc, base_auc):
        """ compute improvement via average
        Args:
            auc1, acu2: np array for tasks
        Returns:
            float: improvements
        """
        improvement =  ((new_auc - base_auc) / new_auc).mean()
        return improvement
    
    
    def _init_files(self, config):
        """
        
        """
        if self.config["resume"]:
            result_path = self.config["resume_path"]
            ckpt_path = os.path.join(result_path, "checkpoints")
            log_path = os.path.join(result_path, "log_files")
            viz_path = os.path.join(log_path, "tfboard_files")
        else:
            base_dir = "{}-{}-{}".format(
                config["model"]["name"],
                config["dataset"].split("/")[-1],
                "{}-{}".format(
                    config["tag"] if config["tag"] is not None else "",
                    get_local_time(),
                )
                if config["log_name"] is None
                else config["log_name"]
            )
            result_path = os.path.join(config["result_path"], base_dir)
            # print("Result DIR: " + result_path)
            ckpt_path = os.path.join(result_path, "checkpoints")
            log_path = os.path.join(result_path, "log_files")
            viz_path = os.path.join(log_path, "tfboard_files")
            if self.rank == 0:
                create_dirs([result_path, log_path, ckpt_path, viz_path])
                with open(
                    os.path.join(result_path, "config.yaml"), "w", encoding="utf-8"
                ) as fout:
                    fout.write(yaml.dump(config))
        return result_path, log_path, ckpt_path, viz_path
    
    
    def _init_device(self, rank, config):
        init_seed(config["seed"], config["deterministic"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_ids"])  # equal to set in shell
        n_gpu = torch.cuda.device_count()
        device = torch.device("cuda:{}".format(self.rank if n_gpu > 0 else "cpu"))
        list_ids = list(range(n_gpu))                                   # rename device ids: 0,1,2,...
        torch.cuda.set_device(self.rank)
        return device, list_ids   
    
    
    def _init_model(self, config):
        net = SuperNet(
            features=self.features,
            embedding_dim=config["embedding_dim"],
            task_types=config["task_types"],
            n_experts=config["model"]["kwargs"]["n_experts"],
            n_expert_layers=config["model"]["kwargs"]["n_expert_layers"],
            n_layers=config["model"]["kwargs"]["expert_module"]["n_layers"],
            in_features=config["model"]["kwargs"]["expert_module"]["in_features"],
            out_features=config["model"]["kwargs"]["expert_module"]["out_features"],
            tower_layers=config["model"]["kwargs"]["tower_layers"],
            dropout=config["model"]["kwargs"]["dropout"],
            expert_candidate_ops=config["model"]["kwargs"]["expert_module"]["ops"],
        )
        net.init_arch_params(init_type="normal", init_ratio=1e-3)
        print(net)
        
        if self.config["resume"]:
            resume_path = os.path.join(
                self.config["resume_path"], "checkpoints", "model_last.pth"
            )
            print("load the resume model checkpoints dict from {}.".format(resume_path))
            net = self._load_model(net, resume_path)
            
        net = net.to(self.rank)
        return net, copy.deepcopy(net.state_dict())
    
    
    def _init_dataloader(self, config): 
        features, x_train, y_train, x_val, y_val, x_test, y_test = self._get_data_dict(config)
        dg = DataGenerator(x_train, y_train)
        train_loader, val_loader, test_loader = dg.generate_dataloader(
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        return train_loader, val_loader, test_loader, features
    
    
    def _init_optim(self, config):
        # weight params
        if config["no_decay_keys"]:
            keys = config["no_decay_keys"].split("#")
            params_dict_list = [
                {
                    "params": self.net.weight_parameters(keys, mode="exclude"), # parameters with weight decay
                },
                {
                    "params": self.net.weight_parameters(keys, mode="include"), # parameters without weight decay
                    "weight_decay": 0,
                }
            ]
        else:
            params_dict_list = [
                {
                    "params": self.net.weight_parameters()
                },
            ]
        optimizer = get_instance(
            torch.optim, "optimizer", config, params=params_dict_list,
        )
        # arch params
        arch_params_dict_list = [
            {
                "params": self.net.architecture_parameters()
            },
        ]
        arch_optimizer = get_instance(
            torch.optim, "arch_optimizer", config, params=arch_params_dict_list,
        )
        scheduler = GradualWarmupScheduler(
            optimizer, self.config
        )  # if config['warmup']==0, scheduler will be a normal lr_scheduler, jump into this class for details
        print(optimizer)
        print(arch_optimizer)
        from_epoch = -1
        best_val_auc = np.zeros(self.task_num)
        if self.config["resume"]:
            resume_path = os.path.join(
                self.config["resume_path"], "checkpoints", "model_last.pth"
            )
            print(
                "load the optimizer, lr_scheduler and epoch checkpoints dict from {}.".format(resume_path)
            )
            all_state_dict = torch.load(resume_path, map_location="cpu")
            state_dict = all_state_dict["optimizer"]
            optimizer.load_state_dict(state_dict)
            state_dict = all_state_dict["arch_optimizer"]
            arch_optimizer.load_state_dict(state_dict)
            state_dict = all_state_dict["lr_scheduler"]
            scheduler.load_state_dict(state_dict)
            from_epoch = all_state_dict["epoch"]
            best_val_auc = all_state_dict["best_val_auc"]
            print("model resume from the epoch {}".format(from_epoch))
        return optimizer, arch_optimizer, scheduler, from_epoch, best_val_auc
    
    
    def _init_logger(self, is_train=True):
        # hack print
        def use_logger(*msg, level="info", file=None):
            try:
                for m in msg:
                    getattr(logger, level)(m)
            except os.error:
                raise ("logging have no {} level".format(level))
        builtins.print = use_logger
        
        filename = "{}-{}-{}.log".format(
            self.config["model"]["name"], "train" if is_train else "test", get_local_time()
        )
        log_path = os.path.join(self.log_path, filename)
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + 
                    "<level>{level: <4}</level> | " + 
                    # "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " + 
                    "<level>{message}</level>",
        )
        logger.add(
            sink=log_path,
            encoding="utf-8",
            enqueue=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + 
                    "<level>{level: <4}</level> | " + 
                    # "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " + 
                    "<level>{message}</level>",
        )
        return logger
    
    
    def _init_writer(self, viz_path):
        """
        Init the tensorboard writer.
        """
        if self.rank == 0:
            writer = TensorboardWriter(viz_path)
            return writer
        else:
            return None
    
    
    def _save_model(self, epoch, save_type=SaveType.NORMAL):
        if save_type == SaveType.NORMAL:
            save_name = os.path.join(self.ckpt_path, "model_{:0>5d}.pth".format(epoch))
            arch_name = os.path.join(self.ckpt_path, "model_{:0>5d}.arch".format(epoch))
        elif save_type == SaveType.BEST:
            save_name = os.path.join(self.ckpt_path, "model_best.pth")
            arch_name = os.path.join(self.ckpt_path, "model_best.arch")
        elif save_type == SaveType.LAST:
            save_name = os.path.join(self.ckpt_path, "model_last.pth")
            arch_name = os.path.join(self.ckpt_path, "model_last.arch")
        
        model_state_dict = OrderedDict(
            {k.replace("module.", ""): v \
                for (k, v) in self.net.state_dict().items()}
        )
        if save_type == SaveType.NORMAL or save_type == SaveType.BEST:
            torch.save(model_state_dict, save_name)
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "arch_optimizer": self.arch_optimizer.state_dict(),
                    "lr_scheduler": self.scheduler.state_dict(),
                    "best_val_auc": self.best_val_auc,
                },
                save_name,
            )
        # save exported arch
        json.dump(self.net.exported_arch, open(arch_name, "w"), indent=4)
    
    
    def _load_model(self, model, model_path, strict=False):
        state_dict = torch.load(model_path, map_location="cpu")["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        if len(msg.missing_keys) != 0:
            print("missing keys:{}".format(msg.missing_keys), level="warning")
        if len(msg.unexpected_keys) != 0:
            print("unexpected keys:{}".format(msg.unexpected_keys), level="warning")
        return model
    
    
    def _cal_time_scheduler(self, start_time, epoch_idx):
        """
        Calculate the remaining time and consuming time of the training process.

        Returns:
            str: A string similar to "00:00:00/0 days, 00:00:00". First: comsuming time; Second: total time.
        """
        total_epoch = self.config["epoch"] - self.from_epoch - 1
        now_epoch = epoch_idx - self.from_epoch

        time_consum = datetime.datetime.now() - datetime.datetime.fromtimestamp(start_time)
        time_consum -= datetime.timedelta(microseconds=time_consum.microseconds)
        time_remain = (time_consum * (total_epoch - now_epoch)) / (now_epoch)
        res_str = str(time_consum) + "/" + str(time_remain + time_consum)
        return res_str
    

    def _get_data_dict(self, config):
        if "dataset_ext" in config and config["dataset_ext"] == "pickle":
            df_train = pd.read_pickle(
                os.path.join(config['dataset_path'], f"{config['dataset']}_train.pkl")
            )
            df_val = pd.read_pickle(
                os.path.join(config['dataset_path'], f"{config['dataset']}_val.pkl")
            )
            df_test = pd.read_pickle(
                os.path.join(config['dataset_path'], f"{config['dataset']}_test.pkl")
            )
            print("pickle")
        else:
            df_train = pd.read_csv(
                os.path.join(config['dataset_path'], f"{config['dataset']}_train.csv")
            )
            df_val = pd.read_csv(
                os.path.join(config['dataset_path'], f"{config['dataset']}_val.csv")
            )
            df_test = pd.read_csv(
                os.path.join(config['dataset_path'], f"{config['dataset']}_test.csv")
            )
        print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))

        train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
        data = pd.concat([df_train, df_val, df_test], axis=0)
        dense_cols = config['dense_fields']
        sparse_cols = config['sparse_fields']
        print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

        # define dense and sparse features
        label_cols = config['label_fields']  # the order of labels can be any
        used_cols = sparse_cols + dense_cols
        features = [
            SparseFeature(col, data[col].max() + 1, embed_dim=config['embedding_dim']) \
                for col in sparse_cols
        ] + [DenseFeature(col) for col in dense_cols]
        x_train, y_train = (
            {name: data[name].values[:train_idx] for name in used_cols}, 
            data[label_cols].values[:train_idx]
        )
        x_val, y_val = (
            {name: data[name].values[train_idx:val_idx] for name in used_cols}, 
            data[label_cols].values[train_idx:val_idx]   
        )
        x_test, y_test = (
            {name: data[name].values[val_idx:] for name in used_cols}, 
            data[label_cols].values[val_idx:]
        )
        return features, x_train, y_train, x_val, y_val, x_test, y_test
    