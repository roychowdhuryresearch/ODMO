import torch
import torch.nn as nn
import torch.optim as optim

import string
import os
import wandb
import time
import copy
import random
import sys
from modules.utils import clean_folder
from modules.trainer import Trainer
from modules.initializer import *
from modules.models import *
from modules.meters import *


def pipeline(args, save_dir, resume_model_path=None):
    device = args["device"]
    dataset_name = args["dataset_name"]
    lr = args["lr"]

    letters = string.ascii_lowercase
    suffix = ''.join(random.choice(letters) for i in range(3))
    tmp_dir = os.path.join(save_dir, dataset_name+ "_" + suffix)
    while os.path.exists(tmp_dir):
        suffix = ''.join(random.choice(letters) for i in range(3))
        tmp_dir = save_dir + "_" + suffix
    save_dir = tmp_dir
    clean_folder(save_dir)

    args["save_dir"] = save_dir
    args_clean = copy.deepcopy(args)
    wandb.config.update(args_clean)

    device = initialize_cuda(device)

    dataloaders = initialize_dataloaders(args)
    model = initialize_model(args)
    if resume_model_path is not None:
        state_dict = torch.load(resume_model_path, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)

    optimizerG = optim.Adam(model.parameters(), lr=lr)
    optimizers = {"generator": optimizerG}

    evaluator = initialize_evaluator(args, classifier_type="a2m")
    trainer = Trainer(
        model, evaluator, optimizers, dataloaders, args, save_dir, device, dataset_name=dataset_name
    )
    return trainer


if __name__ == "__main__":
    
    dataset_name = sys.argv[1]
    device = sys.argv[2]

    model_dir = "ckpt"
    save_dir = os.path.join(model_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    args = initialize_dataset_args(dataset_name)
    args["device"] = device
    run_name = "ODMO_"+ dataset_name
    if not args["use_end"]:
        run_name += "_NE"
    
    wandb.init(project="ODMO", name=run_name)
    resume_model_path = None
    trainer = pipeline(args, save_dir, resume_model_path=resume_model_path)
    trainer.train(100000)
