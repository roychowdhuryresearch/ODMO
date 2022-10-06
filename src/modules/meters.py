import numpy as np
import torch
from torch.utils.data import dataset
from modules.utils import plot_embeddings
import pandas as pd
import plotly.express as px
import param.data_params as data_params
import plotly.graph_objs as go
from modules.initializer import initialize_dataset_args_classnames

class ClassifierMeter:
    def __init__(self):
        self.loss = []
        self.pred = []
        self.label = []

    def add(self, loss, predict, label):
        self.loss.append(loss)
        self.pred.append(predict)
        self.label.append(label)
    def dump(self):
        loss = np.array(self.loss)
        pred = np.concatenate(self.pred, 0)
        label = np.concatenate(self.label, 0)
        return np.mean(loss), pred, label
    

class Meter:
    def __init__(self):
        self.traj_loss = []
        self.motion_loss = []
        self.ctr_loss = []
        self.loss = []
        self.g_loss = []
        self.kld_loss = []

    def add(self, traj_loss, motion_loss, ctr_loss, loss, kld_loss):
        self.traj_loss.append(traj_loss)
        self.motion_loss.append(motion_loss)
        self.ctr_loss.append(ctr_loss)
        self.loss.append(loss)
        self.kld_loss.append(kld_loss)

    def dump(self):
        items = [self.traj_loss, self.motion_loss, self.ctr_loss, self.loss]
        res = []
        for i in items:
            res.append(np.mean(i))
        return np.round(np.array(res), 3)

    def dump_wandb(self):
        items = [
            self.traj_loss,
            self.motion_loss,
            self.ctr_loss,
            self.loss,
            self.kld_loss, 
         ]
        
        names = ["traj_loss", "motion_loss", "ctr_loss", "loss", "kld_loss", "mD_loss"] 
        return {n: np.mean(i) for n, i in zip(names, items)}

class ValidMeter:
    def __init__(self, dataset_name="mocap"):
        self.prediction = []  # GPU and Torch
        self.c_onehot = []
        self.code = []
        self.dataset_name = dataset_name
        _, self.class_info = initialize_dataset_args_classnames(self.dataset_name)

    def add(self, prediction, c_onehot, code):
        self.prediction.append(prediction.detach())
        self.c_onehot.append(c_onehot.detach())
        if code != None:
            self.code.append(code)

    def dump_prediction(self):

        if len(self.prediction) > 1:  # self.prediction is a torch array
            prediction = torch.cat(self.prediction, 0)
            c_onehot = torch.cat(self.c_onehot, 0)
        else:
            prediction = self.prediction[0]
            c_onehot = self.c_onehot[0]

        c_labels_num = torch.argmax(c_onehot, dim=-1)

        return prediction, c_labels_num

    def dump_code(self):
        prediction, c_onehot = self.dump_prediction()
        if len(self.prediction) > 1:  # self.prediction is a torch array
            code = torch.cat(self.code, 0)
        else:
            code = self.code[0]
        return code, c_onehot

    def plot_emb(self, fig):
        embeddings, targets = self.dump_code()
        #print(np.unique(targets.cpu().numpy(), return_counts=True))
        fig = plot_embeddings(
            fig,
            embeddings.cpu().numpy(),
            targets.cpu().numpy(),
            self.class_info,
            xlim=None,
            ylim=None,
            row=1,
            col=1,
        )

    def plot_traj(self, data, topk = 30):
        data = data[:topk]
        color_group = np.tile(
            np.arange(data.shape[0])[:, None], (1, data.shape[1])
        ).reshape(-1)
        # print(color_group)

        data_df = {
            "x": data[:, :, 0].reshape(-1),
            "y": data[:, :, 1].reshape(-1),
            "z": data[:, :, 2].reshape(-1),
            "color": color_group,
        }
        fig = px.line_3d(data_df, x="x", y="y", z="z", color="color")
        return fig

    def plot_traj_end(self, data, pred_end, topk = 30):
        data = data[:topk]
        color_group = np.tile(
            np.arange(data.shape[0])[:, None], (1, data.shape[1])
        ).reshape(-1)

        data_df = {
            "x": data[:, :, 0].reshape(-1),
            "y": data[:, :, 1].reshape(-1),
            "z": data[:, :, 2].reshape(-1),
            "color": color_group,
        }
        fig = px.line_3d(data_df, x="x", y="y", z="z", color="color")

        fig.add_trace(go.Scatter3d(pred_end[:,:,0].reshape(-1),pred_end[:,:,1].reshape(-1), pred_end[:,:,2].reshape(-1),color=color_group, mode='markers'))
        return fig
