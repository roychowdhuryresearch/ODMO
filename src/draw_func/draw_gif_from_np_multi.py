import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from draw_gif_byindex import draw_gif_by_index
from modules.initializer import initialize_dataset_spec

dataset_name = "HumanAct12"
fn = "./sampled_data/HumanAct12/HumanAct12_end_release_MM/GMM_sample_250_0.npz"
out_folder = "./results/gifs"
downsample_rate = 2
n_jobs = 32
os.makedirs(out_folder, exist_ok=True)
loaded = np.load(fn)
class_names, num_joint, motionft_dim, kinematic_tree = initialize_dataset_spec(dataset_name)
motion_arr = loaded["motion"]
class_label = loaded["label"].astype(int)
draw_gif_by_index(motion_arr, class_label, dataset_name, out_folder, downsample_rate, n_jobs)
