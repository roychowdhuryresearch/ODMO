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
from metrics.calculate_metric import run


if __name__ == "__main__":

    os.environ['WANDB_MODE'] = 'offline'
   
    evaluated_data = "HumanAct12_test"
    # the name you want to display on wandb 
    option = None  
    # train or test for uestc dataset for other dataset please put None
    dataset_name = "HumanAct12"
    # select from mocap, HumanAct12, uestc
    folder = "results/endloc_customization/HumanAct12"
    # the path of the folder you want to calculate metric from
    # this folder must have 10 files 
    run(folder, dataset_name, evaluated_data, option)
