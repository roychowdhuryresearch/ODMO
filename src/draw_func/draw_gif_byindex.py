import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import param.data_params as data_params
from draw_func.draw_gif import plot_3d_motion_v2
from tqdm import tqdm
from modules.utils import parallel_process

def initialize_from_dataset(dataset_name):
    if dataset_name == "mocap":
        class_names = data_params.mocap_action_enumerator
        num_joint = 20
        kinematic_tree = data_params.mocap_kinematic_chain
    elif dataset_name == "HumanAct12":
        class_names = data_params.humanact12_class_names
        num_joint = 24
        kinematic_tree = data_params.humanact12_kinematic_chain
    elif dataset_name == "uestc":
        num_joint = 18
        class_names = data_params.uestc_class_names
        kinematic_tree = data_params.uestc_kinematic_chain
    else:
        print("Not implemented")
    
    return num_joint, class_names, kinematic_tree

def general_plot(p, kinematic_tree, fn, interval, dataset_name, num_joint, sample_rate=2):
    plot_3d_motion_v2(
        p.reshape(-1, num_joint, 3)[::sample_rate],
        kinematic_tree,
        fn,
        interval=interval,
        dataset=dataset_name,
    )


def dump_gif(pred, c, folder_name, dataset_name, sample_rate=2, n_jobs=32):
    num_joint, class_names, kinematic_tree = initialize_from_dataset(dataset_name)
    param_list = [{"p":pred[p_idx], 
                   "kinematic_tree":kinematic_tree, 
                   "fn": os.path.join(folder_name,f'{class_names[c[p_idx]]}_pred_{p_idx}.gif'), 
                   "interval":50,
                   "dataset_name":dataset_name,
                   "num_joint":num_joint,
                   "sample_rate":sample_rate } for p_idx in range(len(pred))]
    ret = parallel_process(param_list, general_plot ,n_jobs=32, use_kwargs=True, front_num=3)

def draw_gif_by_index(motion_arr, class_label, dataset_name, out_folder, sample_rate=2, n_jobs=32):
    os.makedirs(out_folder, exist_ok=True)
    dump_gif(motion_arr, class_label, out_folder, dataset_name, sample_rate=sample_rate, n_jobs=n_jobs)

def draw_gif_by_interpolation(pred, class_label, first, second, trials, idx ,dataset_name, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    num_joint, class_names, kinematic_tree = initialize_from_dataset(dataset_name)
    
    param_list = [{"p":pred[p_idx], 
                   "kinematic_tree":kinematic_tree, 
                   "fn":os.path.join(
                    out_folder, f"{class_names[class_label[p_idx]]}_{first[p_idx]}_{second[p_idx]}_{trials[p_idx]}_idx_{idx[p_idx]}.gif"), 
                   "interval":50,
                   "dataset_name":dataset_name,
                   "num_joint":num_joint } for p_idx in range(len(pred))]
    ret = parallel_process(param_list, general_plot ,n_jobs=32, use_kwargs=True, front_num=0)


def draw_gif_by_modes(pred, class_label, modes ,dataset_name, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    num_joint, class_names, kinematic_tree = initialize_from_dataset(dataset_name)
    class_label = class_label.astype(int)
    param_list = [{"p":pred[p_idx], 
                   "kinematic_tree":kinematic_tree, 
                   "fn":os.path.join(
                    out_folder, f"{class_names[class_label[p_idx]]}_{modes[p_idx]}_idx_{p_idx}.gif"), 
                   "interval":50,
                   "dataset_name":dataset_name,
                   "num_joint":num_joint } for p_idx in range(len(pred))]
    ret = parallel_process(param_list, general_plot ,n_jobs=32, use_kwargs=True, front_num=0)

def draw_gif_by_subtype_modes(pred, class_label, subtypes_int, modes, dataset_name, out_folder):
    """
    show the subtype index and clustered index at the same time.
    """
    os.makedirs(out_folder, exist_ok=True)
    num_joint, class_names, kinematic_tree = initialize_from_dataset(dataset_name)
    class_label = class_label.astype(int)
    subtypes_int = subtypes_int.astype(int)
    
    param_list = [{"p":pred[p_idx], 
                   "kinematic_tree":kinematic_tree, 
                   "fn":os.path.join(
                    out_folder, f"{class_names[class_label[p_idx]]}_sub{subtypes_int[p_idx]}_{modes[p_idx]}_idx_{p_idx}.gif"), 
                   "interval":50,
                   "dataset_name":dataset_name,
                   "num_joint":num_joint } for p_idx in range(len(pred))]
    ret = parallel_process(param_list, general_plot ,n_jobs=32, use_kwargs=True, front_num=0)