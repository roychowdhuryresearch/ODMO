import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
import os, wandb, time, copy, sys
from modules.ensemble import Ensemble
from modules.trainer import Trainer
from modules.initializer import *
from modules.models import *
import param.param_mocap
import param.param_humanact12
from inference_odmo import pipeline
    
def find_best_model(model_path):
    model_paths = os.listdir(model_path)
    best_checkpoint_num = 0
    best_checkpoint_path = ""
    for mp in model_paths:
        if "best" not in mp:
            continue
        num = int(mp.split("_")[1].split(".")[0])
        if num > best_checkpoint_num:
            best_checkpoint_path = mp
            best_checkpoint_num = num
    print(best_checkpoint_path)
    return best_checkpoint_path


if __name__ == "__main__":
    save_dir = None
    model_dir = "ckpt"
    model_name = sys.argv[1]
    sampling_strategy = sys.argv[2]
    device = sys.argv[3]
    dataset_name = model_name.split("_")[0]
    sampling_strategy = "MM"
    converted_path = "_".join(model_name.split(".")[0].split("/"))
    converted_path = converted_path + "_" + sampling_strategy

    output_dir = os.path.join("./sampled_data", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    model_folder= os.path.join(model_dir, dataset_name ,model_name)
    model_path = os.path.join(model_folder, find_best_model(model_folder))
    print(model_path)
    args = initialize_dataset_args(dataset_name)
    args["device"] = device
    trainer, train_loader = pipeline(args, save_dir, model_path)

    if dataset_name == "mocap":
        repeat_num = 400
    elif dataset_name == "HumanAct12":
        repeat_num = 250
    else:
        repeat_num = 200

    trainer.predict_npz_samples(
        train_loader,
        repeat_num=repeat_num,
        sampling_strategy= sampling_strategy,
        dump_datadir=os.path.join(output_dir, converted_path)
    )

    print(f'Sample data from {model_name} is accomplished')