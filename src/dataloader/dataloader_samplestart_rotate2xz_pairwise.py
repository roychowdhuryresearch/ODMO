import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder
from modules.utils import create_rotation_vect
from param.data_params import mocap_class_names, humanact12_class_names, uestc_class_names


def create_one_hotft(label, class_name_list):
    encoder = OneHotEncoder(handle_unknown="ignore")
    numeric_label = np.zeros(label.shape)

    for i in range(len(class_name_list)):
        idx = np.where(label == class_name_list[i])
        numeric_label[idx] = i

    encoder.fit(numeric_label.reshape(-1, 1))
    label_new = encoder.transform(numeric_label.reshape(-1, 1)).toarray()
    return label_new

class MotionDataset(Dataset):
    def __init__(self, csv_fn, data_dir, use_lie_group=False, max_length=60):
        clip = (
            pd.read_csv(csv_fn, index_col=False)
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
        self.lengths = []
        self.data = []
        self.labels = []
        self.motion_namelist = []
        self.max_length = max_length
        self.class_name_list = mocap_class_names
        for i in range(clip.shape[0]):
            motion_name = clip.iloc[i]["motion"]
            action_type = clip.iloc[i]["action_type"]
            npy_path = os.path.join(data_dir, motion_name + ".npy")
            # motion_length, joints_num, 3
            pose_raw = np.load(npy_path)
            # rescale the pose
            pose_raw = pose_raw / 20
            # Locate the root joint of initial pose at origin
            # get the offset and return the final pose
            offset_mat = np.tile(pose_raw[0, 0], (pose_raw.shape[1], 1))
            pose_mat = pose_raw - offset_mat
            pose_mat = pose_mat.reshape((-1, 20 * 3))
            self.motion_namelist.append(motion_name)
            self.data.append(pose_mat)
            self.labels.append(action_type)
            self.lengths.append(pose_mat.shape[0])
        print("loaded raw npy data", csv_fn, len(self.data))
        self.preprocess_data()

    def shuffle(self, num_items = 200):
        index = np.arange(len(self.data))
        np.random.shuffle(index)
        return index[:num_items]

    def preprocess_data(self):
        num_sample = len(self.data)
        self.labels = np.array(self.labels)
        self.labels = create_one_hotft(self.labels, self.class_name_list)  ## one hot
        index = np.arange(len(self.data))
        self.data = [self.data[item] for item in index]
        self.labels = [self.labels[item] for item in index]
        self.lengths = [self.lengths[item] for item in index]
        self.motion_namelist = [self.motion_namelist[item] for item in index]
        self.labels_num = np.argmax(self.labels, -1)
        self.pair_mask = self.generate_pair_mask() 
    
    def generate_weights_for_data(self):
        value_key, value_count = np.unique(self.labels_num, return_counts=True)
        num_data = len(self.labels_num)
        class_weights = 1.0/value_count
        base_class_w = np.zeros(num_data)
        for j, class_i in enumerate(value_key):
            class_ids = np.where(self.labels_num == class_i)[0]
            base_class_w[class_ids] = class_weights[j]
        return base_class_w

    def generate_pair_mask(self):
        res = []
        
        data_sampling_w = self.generate_weights_for_data()
        for i in range(np.max(self.labels_num) + 1):
            yes = np.where(self.labels_num == i)[0]
            no = np.where(self.labels_num != i)[0]
            res.append([yes, no, data_sampling_w[no]/6])
        return res

    def convert(self, ind):
        motion = np.array(self.data[ind])
        motion_name = self.motion_namelist[ind]
        label = self.labels[ind]
        length = self.lengths[ind]
        # random sample
        start = 0
        if length >= self.max_length:
            gap = length - self.max_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            end = start + self.max_length
            r_motion = motion[start:end]
            # offset deduction
            r_motion = r_motion - np.tile(
                r_motion[0, :3], (1, int(r_motion.shape[-1] / 3))
            )

        # padding
        else:
            gap = self.max_length - length
            last_pose = np.expand_dims(motion[-1], axis=0)
            pad_poses = np.repeat(last_pose, gap, axis=0)
            r_motion = np.concatenate([motion, pad_poses], axis=0)
        length = min(length, self.max_length)
        return r_motion, label, length, start


    def __getitem__(self, ind):
        r_motion_1, c_onehot_1, length_1, start1  = self.convert(ind)
        num_label = self.labels_num[ind]
        target = np.random.randint(0, 2)
        if target == 1:
            siamese_index = ind
            while siamese_index == ind:
                siamese_index = np.random.choice(self.pair_mask[num_label][0])
        else:
            siamese_index = np.random.choice(self.pair_mask[num_label][1],p=self.pair_mask[num_label][2])
            
        r_motion_2, c_onehot_2, length_2, start2 = self.convert(siamese_index)
        return (r_motion_1, c_onehot_1, length_1), (r_motion_2, c_onehot_2, length_2)
        
    def __len__(self):
        return len(self.data)

class MotionDatasetUestc(data.Dataset):
    def __init__(self, csv_fn, data_dir, use_lie_group=False, max_length=60):
        clip = (
            pd.read_csv(csv_fn, index_col=False)
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
        self.lengths = []
        self.data = []
        self.labels = []
        self.motion_namelist = []
        self.max_length = max_length
        self.class_name_list = uestc_class_names
        for i in range(clip.shape[0]):
            motion_name = clip.iloc[i]["motion"]
            action_type = clip.iloc[i]["action_type"]
            npy_path = os.path.join(data_dir, motion_name.split(".")[0] + ".npy")
            # motion_length, joints_num, 3
            pose_raw = np.load(npy_path)
            # Locate the root joint of initial pose at origin
            offset_mat = np.tile(pose_raw[0, 0], (pose_raw.shape[1], 1))
            pose_mat = pose_raw - offset_mat
            pose_mat = pose_mat.reshape((-1, 18 * 3))
            self.motion_namelist.append(motion_name)
            self.data.append(pose_mat)
            self.labels.append(action_type)
            self.lengths.append(pose_mat.shape[0])
        print("loaded raw npy data", csv_fn, len(self.data))
        self.preprocess_data()

    def preprocess_data(self):
        self.labels = np.array(self.labels)
        self.labels = create_one_hotft(self.labels, self.class_name_list)  ## one hot
        index = np.arange(len(self.data))
        self.data = [self.data[item] for item in index]
        self.labels = [self.labels[item] for item in index]
        self.lengths = [self.lengths[item] for item in index]
        self.motion_namelist = [self.motion_namelist[item] for item in index]
        self.labels_num = np.argmax(self.labels, -1)
        self.pair_mask = self.generate_pair_mask() 
    
    def generate_pair_mask(self):
        res = []
        for i in range(np.max(self.labels_num) + 1):
            yes = np.where(self.labels_num == i)[0]
            no = np.where(self.labels_num != i)[0]
            res.append([yes, no])
        return res

    def convert(self, ind):
        motion = np.array(self.data[ind])
        motion_name = self.motion_namelist[ind]
        label = self.labels[ind]
        length = self.lengths[ind]
        start = 0 
        # random sample
        if length >= self.max_length:
            gap = length - self.max_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            end = start + self.max_length
            r_motion = motion[start:end]
            # offset deduction
            r_motion = r_motion - np.tile(
                r_motion[0, :3], (1, int(r_motion.shape[-1] / 3))
            )
        else:
            gap = self.max_length - length
            last_pose = np.expand_dims(motion[-1], axis=0)
            pad_poses = np.repeat(last_pose, gap, axis=0)
            r_motion = np.concatenate([motion, pad_poses], axis=0)

        length = min(length, self.max_length)
        return r_motion, label, length, start

    def __getitem__(self, ind):
        r_motion_1, c_onehot_1, length_1, start1 = self.convert(ind)
        num_label = self.labels_num[ind]
        target = np.random.randint(0, 2)
        if target == 1:
            siamese_index = ind
            while siamese_index == ind:
                siamese_index = np.random.choice(self.pair_mask[num_label][0])
        else:
            siamese_index = np.random.choice(self.pair_mask[num_label][1])
        
        r_motion_2, c_onehot_2, length_2, start2 = self.convert(siamese_index)
        return (r_motion_1, c_onehot_1, length_1), (r_motion_2, c_onehot_2, length_2)
    def __len__(self):
        return len(self.data)

class MotionDatasetHumanAct12(data.Dataset):
    def __init__(self, csv_fn, data_dir, use_lie_group=False, max_length=60):
        clip = (
            pd.read_csv(csv_fn, index_col=False)
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
        self.lengths = []
        self.data = []
        self.labels = []
        self.motion_namelist = []
        self.max_length = max_length
        self.class_name_list = humanact12_class_names
        for i in range(clip.shape[0]):
            motion_name = clip.iloc[i]["motion"]
            action_type = clip.iloc[i]["action_type"]
            npy_path = os.path.join(data_dir, motion_name + ".npy")
            # motion_length, joints_num, 3
            pose_raw = np.load(npy_path)
            
            # Locate the root joint of initial pose at origin
            offset_mat = np.tile(pose_raw[0, 0], (pose_raw.shape[1], 1))
            pose_mat = pose_raw - offset_mat
            pose_mat = pose_mat.reshape((-1, 24 * 3))
            self.motion_namelist.append(motion_name)
            self.data.append(pose_mat)
            self.labels.append(action_type)
            self.lengths.append(pose_mat.shape[0])
        print("loaded raw npy data", csv_fn, len(self.data))
        self.preprocess_data()
        if use_lie_group:
            self.convert_lie()

    def preprocess_data(self):
        num_sample = len(self.data)
        self.labels = np.array(self.labels)
        self.labels = create_one_hotft(self.labels, self.class_name_list)  ## one hot
        index = np.arange(len(self.data))
        self.data = [self.data[item] for item in index]
        self.labels = [self.labels[item] for item in index]
        self.lengths = [self.lengths[item] for item in index]
        self.motion_namelist = [self.motion_namelist[item] for item in index]
        self.labels_num = np.argmax(self.labels, -1)
        self.pair_mask = self.generate_pair_mask() 
    

    def generate_pair_mask(self):
        res = []
        for i in range(np.max(self.labels_num) + 1):
            yes = np.where(self.labels_num == i)[0]
            no = np.where(self.labels_num != i)[0]
            res.append([yes, no])
        return res

    def convert(self, ind):
        motion = np.array(self.data[ind])
        motion_name = self.motion_namelist[ind]
        label = self.labels[ind]
        length = self.lengths[ind]
        start = 0 
        # random sample
        if length >= self.max_length:
            gap = length - self.max_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            end = start + self.max_length
            r_motion = motion[start:end]
            # offset deduction
            r_motion = r_motion - np.tile(
                r_motion[0, :3], (1, int(r_motion.shape[-1] / 3))
            )

        # padding
        else:
            gap = self.max_length - length
            last_pose = np.expand_dims(motion[-1], axis=0)
            pad_poses = np.repeat(last_pose, gap, axis=0)
            r_motion = np.concatenate([motion, pad_poses], axis=0)

        length = min(length, self.max_length)
        return r_motion, label, length, start

    def __getitem__(self, ind):
        r_motion_1, c_onehot_1, length_1, start1 = self.convert(ind)
        num_label = self.labels_num[ind]
        target = np.random.randint(0, 2)
        if target == 1:
            siamese_index = ind
            while siamese_index == ind:
                siamese_index = np.random.choice(self.pair_mask[num_label][0])
        else:
            siamese_index = np.random.choice(self.pair_mask[num_label][1])
        
        r_motion_2, c_onehot_2, length_2, start2 = self.convert(siamese_index)
        return (r_motion_1, c_onehot_1, length_1), (r_motion_2, c_onehot_2, length_2)
    def __len__(self):
        return len(self.data)


def create_weighted_loaders(
    dataset, batch_size=128,
    seed=0,
    p_val=0.1,
    batch_sample_num = 2048, 
    shuffle=True,
    extras={},
):

    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))
    data_labels = np.argmax(dataset.labels, -1)
    value_key, value_count = np.unique(data_labels, return_counts=True)
    class_weights = 10.0/value_count
    data_weights = np.zeros(data_labels.shape)
    for j, class_i in enumerate(value_key):
        class_id = np.where(data_labels == class_i)[0]
        data_weights[class_id] = class_weights[j]
    
    # Create the validation split from the full dataset
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
        
    val_start = int(np.floor(p_val * dataset_size))
    train_ind = all_indices[val_start:]
    val_ind = all_indices[:val_start]
    if len(val_ind) == 0:
        val_ind = np.array([0])
    remain_ind = list(set(all_indices) - set(val_ind))
    train_weight_list = np.zeros(dataset_size)
    train_weight_list[train_ind] = data_weights[train_ind]

    val_weight_list = np.zeros(dataset_size)
    val_weight_list[val_ind] = data_weights[val_ind]

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    sample_train = WeightedRandomSampler(train_weight_list, batch_sample_num, replacement=True)
    sample_val = WeightedRandomSampler(val_weight_list, 400, replacement=True)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sample_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sample_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    # Return the training, validation, test DataLoader objects
    return train_loader, val_loader
