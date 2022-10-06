

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import param.data_params
import param.param_mocap
import param.param_humanact12
import random
import sys 
from modules.initializer import initialize_dataset_args
from dataloader.dataloader_samplestart_rotate2xz import create_weighted_loaders, MotionDataset, MotionDatasetHumanAct12, MotionDatasetUestc


def dump_gt_data(args, times, save_dir, dataSET, sample_num):

    data_dir = args["data_dir"]
    csv_fn = args["csv_fn"]
    motion_length = args["motion_length"]
    p_val = 0.01
    batch_size = 10000
    dataset = dataSET(csv_fn, data_dir, use_lie_group=False,
                      max_length=motion_length)
    train_loader, val_loader = create_weighted_loaders(
        dataset, batch_size=batch_size, batch_sample_num=batch_size, p_val=p_val, shuffle=True)
    loader = train_loader
    new_motion_list = []
    label_list = []
    for j in range(10):
        for i, (_, motion, label, length) in enumerate(loader, 0):
            new_motion_list.append(motion)
            label_list.append(label)

    motion = torch.cat(new_motion_list)
    label = torch.cat(label_list)
    label_num = torch.argmax(label, -1).detach().numpy()
    num_class = max(label_num) + 1
    motion = motion.detach().numpy()
    print("loaded original data with shape", motion.shape,
          "total num of class is", num_class)

    ft_len = motion.shape[-1]
    motion_arr = np.zeros((sample_num*num_class, motion_length, ft_len))
    label_arr = np.zeros(sample_num*num_class)
    for i in range(num_class):
        data_inds = np.where(label_num == i)[0]
        if len(data_inds) < sample_num:
            print("need more samples for class", i)
        motion_arr[i*sample_num: (i+1) *
                   sample_num] = motion[data_inds[:sample_num]]
        label_arr[i*sample_num: (i+1)*sample_num] = i
        fn = args["dataset_name"] + "_" +  str(motion_length) + "_" + str(sample_num) + "_" + str(times)
        if "test" in args["csv_fn"]:
            fn = args["dataset_name"] + "_test_" + \
                str(motion_length) + "_" + str(sample_num) + "_" + str(times)
    save_dir = os.path.join(save_dir, fn)
    np.savez(save_dir, motion=motion_arr, label=label_arr)
    print("Dumped data!!!!!")

if __name__ == "__main__":
    
    dataset_name = sys.argv[1]

    save_dir = f"./sampled_data/{dataset_name}/real/"
    os.makedirs(save_dir, exist_ok=True)
    
    args = initialize_dataset_args(dataset_name)
    if dataset_name == "mocap":
        sample_num = 400
        dataset = MotionDataset
        for i in range(10):
            dump_gt_data(args, i, save_dir, dataset, sample_num)
    elif dataset_name == "HumanAct12":
        sample_num = 250
        dataset = MotionDatasetHumanAct12
        for i in range(10):
            dump_gt_data(args, i, save_dir, dataset, sample_num)
    elif dataset_name == "uestc":
        dataset = MotionDatasetUestc
        sample_num = 200
        save_dir_train = os.path.join(save_dir,"train")
        os.makedirs(save_dir_train, exist_ok=True)
        for i in range(10):
            dump_gt_data(args, i, save_dir_train, dataset, sample_num)
        args["csv_fn"] = args["test_csv_fn"]
        save_dir_test = os.path.join(save_dir,"test")
        os.makedirs(save_dir_test, exist_ok=True)
        for i in range(10):
            dump_gt_data(args, i, save_dir_test, dataset, sample_num)
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
    
    print(f'Sample real data from {dataset_name} is accomplished')