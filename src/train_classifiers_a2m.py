import torch
import torch.nn as nn
from torch import tensor
import numpy as np
import torch.nn.functional as F
import wandb
import torch.optim as optim
from random import random
from tqdm import tqdm
import os
import copy
import sys
import plotly.figure_factory as ff

import param.data_params as data_params
import param.param_mocap as param_mocap
import param.param_humanact12 as param_humanact12
import param.param_uestc as param_uestc
from dataloader.dataloader_samplestart_rotate2xz import MotionDataset, create_weighted_loaders, MotionDatasetHumanAct12, MotionDatasetUestc


class AMotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size=1, motion_len=None, use_noise=None, device="cpu", classifier_mode="org"):
        super(AMotionDiscriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)
        self.device = device

    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)

        if hidden_unit is None:
            motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(
                motion_sequence.size(1), self.hidden_layer)

        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2, _

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)


class AMotionDiscriminatorForFID(AMotionDiscriminator):
    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(
                motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        return lin1


def extract_localmov_traj(src):
    feature_len = src.shape[2]
    src1 = (src - src[:, :, :3].repeat(1, 1,
            int(feature_len / 3))).clone().detach()
    traj = src[:, :, :3].clone().detach()
    return src1, traj


def train(model, iterator, criterion, optimizer, text_label, device):
    model.train()
    loss_list = []
    for mb, (_, data, label, length) in enumerate(iterator, 0):
        optimizer.zero_grad()
        data, label, length = (
            data.to(device).float(),
            label.to(device).float(),
            length.to(device).float(),
        )
        pred_label, _ = model(data)
        label_num = torch.argmax(label, -1)
        loss = criterion(pred_label, label_num)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    confusion_np, acc, fig = cal_model_acc(
        torch.argmax(pred_label, -1), label_num, text_label)
    return sum(loss_list) / len(loss_list), acc


def validation(model, iterator, criterion, text_label, device, dump=False):
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for _, (_, data, label, length) in enumerate(iterator, 0):
            data, label, length = (
                data.to(device).float(),
                label.to(device).float(),
                length.to(device).float(),
            )
            pred_label, _ = model(data)
            label_num = torch.argmax(label, -1)
            loss = criterion(pred_label, label_num)
            valid_loss.append(loss.item())
    confusion_np, acc, fig = cal_model_acc(
        torch.argmax(pred_label, -1), label_num, text_label)
    return sum(valid_loss) / len(valid_loss), confusion_np, acc, fig


def cal_model_acc(pred_label, label, text_label):
    confusion = torch.zeros(len(text_label), len(text_label), dtype=torch.long)

    for label, pred in zip(label, pred_label):
        confusion[label][pred] += 1
    confusion_np = confusion.cpu().numpy()
    acc = np.sum(np.diag(confusion_np)) / np.sum(confusion_np)

    z_text = [[str(y) for y in x] for x in confusion_np]
    fig = ff.create_annotated_heatmap(
        confusion_np, x=text_label, y=text_label, annotation_text=z_text, colorscale='blues')

    return confusion_np, acc, fig


if __name__ == "__main__":
    wandb.init(project="Motion_classifier")

    model_save_dir = "./ckpt/classifier"

    dataset_name = sys.argv[1]
    device = sys.argv[2]
    
    if dataset_name == "mocap":
        args = param_mocap.param_dict
        DataSET = MotionDataset
        text_label = data_params.mocap_class_names
    elif dataset_name == "HumanAct12":
        args = param_humanact12.param_dict
        DataSET = MotionDatasetHumanAct12
        text_label = data_params.humanact12_class_names
    elif dataset_name == "uestc":
        args = param_uestc.param_dict
        DataSET = MotionDatasetUestc
        text_label = data_params.uestc_class_names

    epoch = 15000
    data_dir = args["data_dir"]
    csv_fn = args["csv_fn"]
    lr = 1e-3

    args["classifier_mode"] = "traj_mv"
    classifier_mode = args["classifier_mode"]
    output_size = args["num_class"]
    layers = args["c_layers"]
    hidden_dim = args["c_hidden_dim"]
    if classifier_mode == "traj_mv":
        input_size = 3
    else:
        input_size = args["motionft_dim"]

    motion_len = args["motion_length"]

    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        device = torch.device(device)
        extras = {"num_workers": 0, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    dataset = DataSET(csv_fn, data_dir, input_size, max_length=motion_len)
    train_loader, val_loader = create_weighted_loaders(
        dataset, batch_size=2048, batch_sample_num=2048, p_val=0.1, extras=extras)
    print("length of train data loader", len(train_loader))

    model = AMotionDiscriminator(
        input_size,
        hidden_dim,
        layers,
        motion_len=motion_len,
        output_size=output_size,
        use_noise=None,
        device=device,
        classifier_mode=classifier_mode
    )

    model = model.to(device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    stats = np.zeros(10)
    best_loss = 100
    fig = None

    model_final_savedir = os.path.join(
        model_save_dir, "ckpt", "classifier", dataset_name, classifier_mode)
    os.makedirs(model_final_savedir, exist_ok=True)
    args["model_save_dir"] = model_final_savedir
    wandb.config.update(args)
    for e in tqdm(range(epoch)):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, text_label, device)
        if e % 100 == 0:
            valid_loss, confusion_np, acc, fig = validation(
                model, val_loader, criterion, text_label, device
            )
        log_dict = {
            "train_acc": round(train_acc, 3),
            "trainL:": round(train_loss, 3),
            "valiL": round(valid_loss, 3),
            "confusion_mat": fig,
            "val_acc": acc,
        }
        wandb.log(log_dict)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(model)
        if e % 500 == (500 - 1):
            torch.save(model.state_dict(), f"{model_final_savedir}/{e}.pkg")
    model.eval()
