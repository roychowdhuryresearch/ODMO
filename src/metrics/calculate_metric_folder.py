import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import torch
import numpy as np
import plotly.figure_factory as ff
import pickle
import wandb
import random
from statistics import NormalDist

from modules.initializer import initialize_evaluator, initialize_dataset_args_classnames


class Cumulater:
    def __init__(self):
        self.acc = []
        self.multim = []
        self.diver = []
        self.fid = []
        self.joint_var = []
        self.loss = []

    def add(self, acc, multim, diver, fid, loss):
        self.acc.append(acc)
        self.multim.append(multim)
        self.diver.append(diver)
        self.fid.append(fid)
        self.loss.append(loss)

    def dump(self):
        acc = np.array(self.acc)
        multim = np.array(self.multim)
        diver = np.array(self.diver)
        fid = np.array(self.fid)
        loss = np.array(self.loss)
        return acc, multim, diver, fid, loss


def load_data(data_path):
    loaded = np.load(data_path)
    motion, label = loaded["motion"], loaded["label"]
    return motion, label


def pipeline(data_path, args):
    device = "cpu"
    args["device"] = device  # we calculate it in CPU here
    evaluator = initialize_evaluator(args)
    cumulater = Cumulater()
    fns = os.listdir(data_path)
    for fn in sorted(fns):
        motion, label = load_data(os.path.join(data_path, fn))
        label = label.astype(int)
        res_dict = evaluator.evaluate_metrics(
            torch.from_numpy(motion), torch.from_numpy(label))
        cumulater.add(
            res_dict["Accuracy"],
            res_dict["Multimodality"],
            res_dict["Diversity"],
            res_dict["FID"],
            res_dict["class_entropy"],
        )
    accuracy, multimodality, diversity, fid, loss = cumulater.dump()

    stats_dict = {
        "accuracy": accuracy,
        "multimodality": multimodality,
        "diversity": diversity,
        "fid": fid,
        "loss": loss,
        "Confusion": res_dict["Confusion"]
    }
    return stats_dict


def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    return h


if __name__ == "__main__":

    os.environ['WANDB_MODE'] = 'offline'
    
    evaluated_data = sys.argv[1]
    option = None
    if len(sys.argv) == 3:
        option = sys.argv[2]
    
    dataset_name = evaluated_data.split("_")[0]
    folder = os.path.join(f"sampled_data/{dataset_name}",evaluated_data)

    args, class_names = initialize_dataset_args_classnames(dataset_name)
    
    args["option"] = option
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    classifier_mode = "org"
    
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    project_name = "Metric_" + dataset_name
    run = wandb.init(project=project_name, entity="mg")
    run_name = evaluated_data+"_"+classifier_mode
    wandb.run.name = run_name
    wandb.run.save()
    wandb.config.update({"dataset_name": dataset_name})
    res_dict = pipeline(folder, args)
    z_text = [[str(y) for y in x] for x in res_dict["Confusion"]]
    fig2 = ff.create_annotated_heatmap(
        res_dict["Confusion"], x=class_names, y=class_names, annotation_text=z_text, colorscale='blues')

    wandb_log = {
        "acc_mean": np.mean(res_dict["accuracy"]),
        "acc_conf": confidence_interval(res_dict["accuracy"]),
        "multimodality_mean": np.mean(res_dict["multimodality"]),
        "multimodality_conf": confidence_interval(res_dict["multimodality"]),
        "fid_mean": np.mean(res_dict["fid"]),
        "fid_conf": confidence_interval(res_dict["fid"]),
        "diversity_mean": np.mean(res_dict["diversity"]),
        "diversity_conf": confidence_interval(res_dict["diversity"]),
        "confusion_fig": fig2
    }
    wandb.log(wandb_log)
    run.finish()
