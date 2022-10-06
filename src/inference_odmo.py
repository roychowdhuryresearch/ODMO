import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample

import os, wandb, time, copy
from modules.trainer import Trainer
from modules.initializer import *
from modules.models import *
import sys

def pipeline(args, save_dir, model_path):
    args["save_dir"] = save_dir
    
    device = args["device"]
    dataset_name = args["dataset_name"]
    lr = args["lr"]

    device = initialize_cuda(device)
    dataloaders = initialize_dataloaders(args)
    model = initialize_model(args)
    model = model.to(device)
    loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(loaded)
    print("model loaded", model_path)

    optimizerG = optim.Adam(model.parameters(), lr=lr)
    optimizers = {"generator": optimizerG}
    evaluator = initialize_evaluator(args, classifier_type="a2m")
    trainer = Trainer(
        model, evaluator, optimizers, dataloaders, args, save_dir, device, dataset_name=dataset_name
    )
    return trainer, dataloaders["train"]
    


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = None
    model_dir = "ckpt/pretrained"
    dataset_name = sys.argv[1]
    sampling_strategy = sys.argv[2]
    use_end = (sys.argv[3] == "use_end")
    device = sys.argv[4]
    if dataset_name == "mocap":
        if use_end:
            model_name = "mocap_end_release.pkg"
        else:
            model_name = "mocap_noend_release.pkg"
    elif dataset_name == "HumanAct12":
        if use_end:
            model_name = "HumanAct12_end_release.pkg" 
        else:
            model_name = "HumanAct12_noend_release.pkg"
    elif  dataset_name == "uestc":
        if use_end:
            model_name = "uestc_end_release.pkg"
        else:
            model_name = "uestc_noend_release.pkg"
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
    
    dataset_name = model_name.split("_")[0]
    converted_path = "_".join(model_name.split(".")[0].split("/"))
    converted_path = converted_path + "_" + sampling_strategy

    output_dir = os.path.join("sampled_data", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(model_dir, model_name)
    args = initialize_dataset_args(dataset_name)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    args["device"] = device
    args["use_end"] = use_end
    trainer, train_loader = pipeline(args, save_dir, model_path)
    dump_data = True
    repeat_num = args["repeat_num"]
    if dataset_name == "mocap":
        num_itr = 5
    elif dataset_name == "HumanAct12":
        num_itr = 15
    else:
        num_itr = 8
        
    trainer.predict_npz_samples(
        train_loader,
        repeat_num=repeat_num,
        sampling_strategy= sampling_strategy,
        num_itr = num_itr,
        dump_datadir=os.path.join(output_dir, converted_path)
    )

    print(f'Sample data from {model_name} is accomplished')