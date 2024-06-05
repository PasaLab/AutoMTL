# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

sys.dont_write_bytecode = True
PROJ_PATH = Path(__file__).parent.parent.as_posix()
sys.path.append(PROJ_PATH)
print(sys.path)

from configs.config import Config
from src.trainer.searcher import ArchSearchRunManager


VAR_DICT = {
    # "dataset_name": "UserBehavior",
    # "dataset_name": "ijcai15",
    "dataset_name": "KuaiRand",
    # "dataset_name": "QB-Video",
    # "dataset_name": "Ali-CCP",
    "num_workers": 8,
    "earlystop_patience": 3,
    "device_ids": 0,
    "batch_size": 2048,
    
    "epoch": 15, 
    "warmup_epochs": 2,
    "full_arch_train_epochs": 3,
    "fine_tune_epochs": 5,
    "discretize_ops": 10,
    "test_epoch": 2,
    
    "model": {
        "name": "SuperNet",
        "kwargs": {
            "dropout": 0.2,
            "tower_layers": [64],
            # search space
            "n_expert_layers": 2,
            "n_experts": 4,
            "expert_module": {
                "in_features": 64,
                "out_features": 64,
                "n_layers": 3,
                "ops": [
                    "Identity", "MLP-16", "MLP-32", "MLP-64",
                    "MLP-128", "MLP-256", "MLP-512", "MLP-1024",
                ]
            }
        }
    },
    
}


def main():
    """
    console_params > run_trainer.py dict > user defined yaml > default.yaml
    """
    config = Config("./configs/default_nas.yaml", VAR_DICT).get_config_dict()
    if "dataset_name" in config:    # config dataset via console args 'dataset_name'
        config["dataset"]       = DATASET_COLLECTION[config["dataset_name"]]["dataset"]
        config["dataset_path"]  = DATASET_COLLECTION[config["dataset_name"]]["dataset_path"]
        config["dataset_ext"]   = DATASET_COLLECTION[config["dataset_name"]]["dataset_ext"]
        config["dense_fields"]  = DATASET_COLLECTION[config["dataset_name"]]["dense_fields"]
        config["sparse_fields"] = DATASET_COLLECTION[config["dataset_name"]]["sparse_fields"]
        config["label_fields"]  = DATASET_COLLECTION[config["dataset_name"]]["label_fields"]
        config["task_types"]    = DATASET_COLLECTION[config["dataset_name"]]["task_types"]
        config["embedding_dim"] = DATASET_COLLECTION[config["dataset_name"]]["embedding_dim"]
        config["criterions"]    = DATASET_COLLECTION[config["dataset_name"]]["criterions"]
        config["val_metrics"]   = DATASET_COLLECTION[config["dataset_name"]]["val_metrics"]

    if config["hpo_tune"]:
        import nni
        tune_params = nni.get_next_parameter()
        print(tune_params)
        config["optimizer"]["kwargs"]["lr"] = tune_params["lr"]
        
    searcher = ArchSearchRunManager(config, rank=0)
    searcher.fit()
    auc = searcher.evaluate()
    print(f"Final auc.mean: {auc.mean()}")
    
    if config["hpo_tune"]:
        nni.report_final_result(auc.mean())



DATASET_COLLECTION = {
    "UserBehavior": {
        "dataset"       : "UserBehavior",
        "dataset_path"  : "/data/datasets/UserBehavior/",  # data/datasets/UserBehavior
        "dataset_ext"   : "csv",
        "dense_fields"  : [],
        "sparse_fields" : ["user_id:token", "item_id:token", "category:token"],
        "label_fields"  : ["click:label", "buy:label", "cart:label", "favourite:label"],
        "task_types"    : ["classification", "classification", "classification", "classification"],
        "embedding_dim" : 16,
        "criterions"    : ["bce", "bce", "bce", "bce"],
        "val_metrics"   : ["auc", "auc", "auc", "auc"],
    },
    "ijcai15": {
        "dataset"       : "ijcai15",
        "dataset_path"  : "/data/datasets/ijcai15/",       # data/datasets/ijcai15
        "dataset_ext"   : "csv",
        "dense_fields"  : [],
        "sparse_fields" : ["user_id:token", "item_id:token", "cat_id:token", "seller_id:token", "brand_id:token", "age_range:token", "gender:token"],
        "label_fields"  : ["purchase:label", "favourite:label"],
        "task_types"    : ["classification", "classification"],
        "embedding_dim" : 16,
        "criterions"    : ["bce", "bce"],
        "val_metrics"   : ["auc", "auc"],
    },
    "KuaiRand": {
        "dataset"       : "KuaiRand",
        "dataset_path"  : "/data/datasets/KuaiRand/",      # data/datasets/KuaiRand
        "dataset_ext"   : "csv",
        "dense_fields"  : [],
        "sparse_fields" : ["user_id", "video_id", "is_rand", "tab", "user_active_degree", "is_lowactive_period", 
                           "is_live_streamer", "is_video_author", "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range", 
                           "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3", "onehot_feat4", "onehot_feat5", 
                           "onehot_feat6", "onehot_feat7", "onehot_feat8", "onehot_feat9", "onehot_feat10", "onehot_feat11", 
                           "onehot_feat12", "onehot_feat13", "onehot_feat14", "onehot_feat15", "onehot_feat16", "onehot_feat17", 
                           "author_id", "video_type", "music_id", "music_type"],
        "label_fields"  : ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view"],
        "task_types"    : ["classification", "classification", "classification", "classification", "classification", "classification", "classification"],
        "embedding_dim" : 16,
        "criterions"    : ["bce", "bce", "bce", "bce", "bce", "bce", "bce"],
        "val_metrics"   : ["auc", "auc", "auc", "auc", "auc", "auc", "auc"],
    },
    "QB-Video": {
        "dataset"       : "QB_Video",
        "dataset_path"  : "/data/datasets/QB-Video/",      # data/datasets/QB-Video
        "dataset_ext"   : "csv",
        "dense_fields"  : [],
        "sparse_fields" : ["user_id", "item_id", "video_category", "watching_times", "gender", "age"],
        "label_fields"  : ["click", "follow", "like", "share"],
        "task_types"    : ["classification", "classification", "classification", "classification"],
        "embedding_dim" : 16,
        "criterions"    : ["bce", "bce", "bce", "bce"],
        "val_metrics"   : ["auc", "auc", "auc", "auc"],
    },
    "Ali-CCP": {
        "dataset"       : "ali_ccp",
        "dataset_path"  : "/data/datasets/Ali-CCP/",       # data/datasets/Ali-CCP
        "dataset_ext"   : "pickle",
        "dense_fields"  : ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"],
        "sparse_fields" : ["101", "121", "122", "124", "125", "126", "127", "128", "129", "205", "206", "207", "210", "216", "508", "509", "702", "853", "301", "109_14", "110_14", "127_14", "150_14"],
        "label_fields"  : ["click", "purchase"],
        "task_types"    : ["classification", "classification"],
        "embedding_dim" : 16,
        "criterions"    : ["bce", "bce"],
        "val_metrics"   : ["auc", "auc"],
    },
}


if __name__ == "__main__":
    main()