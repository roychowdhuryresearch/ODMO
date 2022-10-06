import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import torch
import numpy as np
from modules.initializer import initialize_evaluator, initialize_dataset_args_classnames
from metrics.calculate_metric import confidence_interval, load_data
from sklearn.metrics import pairwise_distances

def compute_APD(data):
    # data shape (n_seq, n_feature)
    num_data = data.shape[0]
    dist = pairwise_distances(data, data)
    APD = np.sum(dist)
    if len(dist) > 1:
        APD = APD/(num_data*(num_data - 1))
    return dist, APD

def compute_mode_APD(data_xyz, modes_index):
    apd_list = []
    for i in range(max(modes_index)+ 1):
        ids = np.where(modes_index==i)[0]
        sel_data = data_xyz[ids]
        sel_data_xyz = sel_data.reshape(sel_data.shape[0], -1)
        _, apd = compute_APD(sel_data_xyz)
        apd_list.append(apd)
    return apd_list

def pipeline(data_path):
    repeat_num = 10
    class_apd = []
    fns = os.listdir(data_path)
    for fn in sorted(fns):
        motion, label = load_data(os.path.join(data_path, fn))
        label = label.astype(int)
        class_apd.append(compute_mode_APD(motion, label))
    return np.array(class_apd)

def run(evaluated_data, option=None):
    dataset_name = evaluated_data.split("_")[0]
    dataset_folder = os.path.join(f"./sampled_data/{dataset_name}")
    model_modes = []

    real_folder = os.path.join(dataset_folder,"real")
    if dataset_name == "uestc":
        if option == "test":
            real_folder = os.path.join(real_folder,"test")
        else:
            real_folder = os.path.join(real_folder,"train")

    _, class_names = initialize_dataset_args_classnames(dataset_name)
    model_modes.append(real_folder)
    model_folder = os.path.join(dataset_folder,evaluated_data)    
    model_modes.append(model_folder)
    res = {}
    for i, data_path in enumerate(model_modes):
        stats = pipeline(data_path)
        if i == 0:
            real = stats
        else:
            ss = np.mean(stats / real, -1)
            res[dataset_name] = [np.mean(ss), confidence_interval(ss)]
    return res

if __name__ == "__main__":
    evaluated_data = sys.argv[1]
    option = None
    if len(sys.argv)  == 3:
        option = sys.argv[2]
    
    res = run(evaluated_data, option)
    for r in res:
        v = res[r]
        print(r, "$\mstd{" + str(np.round(v[0]*100, 2)) + "}{"+str(np.round(v[1]*100, 2)) + "}$")


