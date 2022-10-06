
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import torch
import torch.nn as nn
import random

import matplotlib.pyplot as plt
import seaborn as sns

from modules.initializer import *
from modules.models import *
from modules.meters import *
from inference_odmo import pipeline
from inference_odmo_best import find_best_model

def draw_trajectory(ends, trajectory, fn):
    plt.figure()
    for i in range(len(trajectory)):
        t = trajectory[i]
        plt.scatter(ends[i, 0], ends[i, 2])
        plt.plot(t[:, 0], t[:, -1])
    plt.axis('square')
    plt.savefig(fn)
    plt.close()
def compare_trajectory(pred_traj, trajectory, fn):
    plt.figure()
    p = plt.get_cmap('tab10')
    for i in range(len(trajectory)):
        t = trajectory[i]
        tt = pred_traj[i]
        plt.plot(tt[:, 0], tt[:, 2], "--", linewidth=2, color=p(i))
        plt.plot(t[:, 0], t[:, -1], color=p(i))
    plt.axis('square')
    plt.savefig(fn)
    plt.close()

def compute_control_error(end, pred_end, threshold=0.1):
    """
    Compute MSE error and error rate based on torelance 
    """
    num_data = end.shape[0]
    sq_error = np.sum((end-pred_end)**2, axis=-1)
    tolerant_upper = np.sum(end**2, axis=-1)*threshold**2
    mse_error = np.mean(sq_error, axis=-1)
    error_rate = np.sum(sq_error < tolerant_upper)/ num_data
    return mse_error, error_rate


def endloc_customization(trainer, save_dir, draw_traj=True, draw_gif= False, compute_metric=False):
    model = trainer.model
    model.enable_eval()
    num_class = model.tgenerator.class_dim

    GMMs, mapping, gt_code_list, gt_end_list = model.create_GMM(trainer.train_loader, num_class) 
    n_sample = 10
    motion_list, label_list = [], []
    class_mse = []
    class_error_rate=[]
    for c in range(num_class):
        (s_1, _) = next(iter(trainer.train_loader)) #dummpy
        data = s_1[0][:n_sample].float().to(trainer.device)
        c_onehot = torch.zeros(n_sample, num_class)
        end_all, pred_end_all = [], []
        with torch.no_grad():
            for i in range(40):
                index = np.arange(len(gt_code_list[c]))
                source_index = np.random.choice(index)
                end_index = np.random.choice(index)
                end1 = gt_end_list[c][source_index]
                end2 = gt_end_list[c][end_index]
                sampled_code = gt_code_list[c][source_index].reshape(1, -1)
                code = torch.zeros((n_sample, sampled_code.shape[1])) 
                code[:] = torch.from_numpy(sampled_code)
                c_onehot[:, c] = 1
                end = torch.from_numpy(np.linspace(tuple(end1), tuple(end2), n_sample))

                code, c_onehot, end = (
                    code.to(trainer.model.device).float(),
                    c_onehot.to(trainer.model.device).float(),
                    end.to(trainer.model.device).float()
                )
                pred_traj = trainer.model.tgenerator(code, c_onehot, end)
                pred_traj = trainer.model.tgenerator.to_location(pred_traj)
                pred_motion = trainer.model.mgenerator(data, c_onehot, code, pred_traj, tf_rate=0, gt_traj=pred_traj, fix_bonelen=True).cpu().numpy()
                pred_end = trainer.model.extract_localmov_traj(torch.from_numpy(pred_motion))[1][:, -1, :]
                end_all.append(end.cpu().numpy())
                pred_end_all.append(pred_end.cpu().numpy())
                class_label = np.argmax(c_onehot.cpu().numpy(), -1)
                motion_list.append(pred_motion)
                label_list.append(class_label)
            if compute_metric: 
                mse, error_rate = compute_control_error(np.concatenate(end_all),np.concatenate(pred_end_all), threshold=0.1)
                class_mse.append(mse)
                class_error_rate.append(error_rate)
    np.savez_compressed(os.path.join(save_dir + ".npz"), motion = np.concatenate(motion_list), label= np.concatenate(label_list) ) 
    print("avg error and error rate", np.mean(class_mse), np.mean(class_error_rate))
    return np.mean(class_mse)

def run(seed, model_name_list, device):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    draw_gif = False
    draw_traj = False
    compute_metric = True
    res = {}
    for model_name in model_name_list:
        dataset_name = model_name.split("_")[0]
        save_dir = os.path.join("./results/endloc_customization", dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        args = initialize_dataset_args(dataset_name)
        args["device"] = device
        trainer, train_loader = pipeline(args, save_dir, model_path)
        print(dataset_name)
        save_dir = os.path.join(save_dir,f"{temp_name}_" + str(seed))
        mse = endloc_customization(trainer, save_dir, draw_traj=draw_traj, draw_gif=draw_gif, compute_metric=compute_metric)
        res[model_name] = mse
    res["seed"] = seed
    return res
if __name__ == "__main__":
    # mocap 54   
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset_name = sys.argv[1]
    pretrained = sys.argv[2] == 'pretrained'
    device = sys.argv[3]
    
    if pretrained:
        model_dir = "ckpt/pretrained"
        model_name = f"{dataset_name}_end_release.pkg"
    else:
        model_dir = os.path.join("ckpt", os.path.join(sys.argv[2]))
        model_name = find_best_model(model_dir)
    temp_name = model_name.split(".")[0]
    df = pd.DataFrame()
    for seed in range(10):
        sed_res = run(seed, [model_name],device)
        df = df.append(pd.DataFrame.from_records([sed_res], index='seed'))
    df.to_csv(os.path.join("./results/endloc_customization",f"end_loc_metric_{temp_name}.csv"))