import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns

from modules.utils import clean_folder
from modules.initializer import *
from modules.models import *
from modules.meters import *
from inference_odmo import pipeline
from inference_odmo_best import find_best_model
from draw_func.draw_gif_byindex import draw_gif_by_interpolation, initialize_from_dataset
 
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


## overwrite sample code multimodal to just GMM the code for each class and get gmm prediction
def cluster_multimodal(code, return_model = False):

    class_code = np.unique(code, axis = 0)
    s_pre = -1
    K = np.arange(3,10)
    for i, k in enumerate(K):
        model = GaussianMixture(n_components=k, n_init=5, init_params='kmeans', random_state=0)
        labels = model.fit_predict(class_code)
        s = metrics.silhouette_score(class_code, labels, metric='euclidean')
        if s < s_pre * 0.95:
            break
        s_pre = s
    num_multimodality = k - 1
    gm = GaussianMixture(n_components=num_multimodality, n_init=5, init_params='kmeans', random_state=0).fit(class_code)
    #print("num_multimodality", num_multimodality)
    if return_model:
        return code, gm.predict(code), gm
    else:
        return code, gm.predict(code)

def draw_multimodel(code, mode, class_name, save_dir):
    num_joint, class_names, kinematic_tree = initialize_from_dataset(dataset_name)
    #print(code.shape)
    X_embedded = TSNE(n_components=2,init='pca').fit_transform(code)

    fig, ax = plt.subplots(figsize=(8, 8))
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.scatterplot(x = X_embedded[:,0], y = X_embedded[:,1], hue = mode[:].astype(int),  palette="tab10", legend="brief", ax = ax)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.legend(ncol=2)
    ax.set_title(class_name)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.savefig(os.path.join(save_dir, str(class_name)+"_"+class_names[class_name] + ".jpg"))
    return X_embedded, mode

def interpolate(trainer, gm, code, class_index, gt_end ,num_trials, num_interpolation):
    (s_1, _) = next(iter(trainer.train_loader)) #dummpy
    data = s_1[0][:num_interpolation].float().to(trainer.device)
    pred_motion_list, code_list, c_onehot_list, first, second, trials, idx = [], [], [], [],[], [], []
    mode_unique = np.arange(len(gm.means_))
    for t in range(num_trials):
        for i in mode_unique:
            for j in mode_unique:
                if i == j:
                    continue
                code1 = np.random.multivariate_normal(gm.means_[i], gm.covariances_[i], 1).squeeze()
                code2 = np.random.multivariate_normal(gm.means_[j], gm.covariances_[j], 1).squeeze()
                
                sample_code = torch.from_numpy(np.linspace(tuple(code1), tuple(code2), num_interpolation))
                end = trainer.model.find_closest_end(sample_code, torch.from_numpy(code), torch.from_numpy(gt_end))

                c_onehot = torch.from_numpy(np.zeros(trainer.model.tgenerator.class_dim)).repeat(len(sample_code), 1)
                c_onehot[:, class_index] = 1
                sample_code, c_onehot, end = (
                    sample_code.to(trainer.device).float(),
                    c_onehot.to(trainer.device).float(),
                    end.to(trainer.device).float()
                )
                pred_traj = trainer.model.tgenerator(sample_code, c_onehot, end)
                pred_traj = trainer.model.tgenerator.to_location(pred_traj)
                pred_motion = trainer.model.mgenerator(
                    data, c_onehot, sample_code, pred_traj, tf_rate=0, gt_traj=pred_traj, fix_bonelen=True
                )
                pred_motion = pred_motion.detach().cpu().numpy()
                pred_motion_list.append(pred_motion)
                code_list.append(sample_code.cpu().numpy())
                c_onehot_list.append([class_index]*num_interpolation)
                first.append([i]*num_interpolation)
                second.append([j]*num_interpolation)
                trials.append([t]*num_interpolation)
                idx.append(np.arange(num_interpolation))
    return pred_motion_list, code_list, c_onehot_list, first, second, trials, idx


def modes_discovery(trainer, save_dir, num_trials = 5, num_interpolation = 7, num_itr=15 ,plot_gif = False):
    model = trainer.model
    model.enable_eval()
    num_class = model.tgenerator.class_dim
    (s_1, _) = next(iter(trainer.train_loader)) #dummpy
    GMMs, mapping, gt_code_list, gt_end_list  = model.create_GMM(trainer.train_loader, num_class, num_itr= num_itr)
    
    pred_motion_list, code_list, c_onehot_list, first, second, trials, idx, embed, embed_modes  = [], [], [], [],[], [], [], [], []
    with torch.no_grad():
        # class iter
        for i in range(num_class):
            code, mode_index, gm = cluster_multimodal(gt_code_list[i], return_model= True)
            
            res_emb = draw_multimodel(code, mode_index, i, save_dir)  
            res = interpolate(trainer, gm, code, i, gt_end_list[i], num_trials, num_interpolation)
            pred_motion_list.extend(res[0])
            code_list.extend(res[1])
            c_onehot_list.extend(res[2])
            first.extend(res[3])
            second.extend(res[4])
            trials.extend(res[5]) 
            idx.extend(res[6])
            embed.append(res_emb[0])
            embed_modes.append(res_emb[1])    

    pred_motion = np.concatenate(pred_motion_list)
    c_label_num = np.concatenate(c_onehot_list)
    first = np.concatenate(first)
    second = np.concatenate(second)
    trials = np.concatenate(trials)
    idx = np.concatenate(idx)
    embed = np.array(embed) 
    embed_modes = np.array(embed_modes)
    np.savez(os.path.join(save_dir, "mode_discovery.npz"), embed = embed, embed_modes = embed_modes ,motion = pred_motion, label = c_label_num, first = first, second = second, trials = trials, idx = idx)
    if plot_gif:
        gif_dir = os.path.join(save_dir, "gifs")
        clean_folder(gif_dir)
        draw_gif_by_interpolation(pred_motion, c_label_num, first, second, trials, idx, dataset_name, gif_dir)


if __name__ == "__main__":
    
    model_dir = "ckpt"

    plot_gif = False
    dataset_name = sys.argv[1]
    pretrained = sys.argv[2] == 'pretrained'
    use_end = (sys.argv[3] == "use_end")
    device = sys.argv[4]
    if pretrained:
        model_dir = "ckpt/pretrained"
        if use_end:
            model_name = f"{dataset_name}_end_release.pkg"
        else:
            model_name = f"{dataset_name}_noend_release.pkg"
    else:
        model_dir = os.path.join("ckpt", os.path.join(sys.argv[2]))
        model_name = find_best_model(model_dir)

    if dataset_name == "mocap":
        num_itr=10
    elif dataset_name == "HumanAct12":
        num_itr=15
    elif dataset_name == "uestc":
        num_itr = 8
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
    dataset_name = model_name.split("_")[0]
    save_dir = os.path.join("./results/modes_discovery", dataset_name)
    if use_end:
        save_dir += "_use_end"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(model_dir, model_name)
    args = initialize_dataset_args(dataset_name)
    args["device"] = device
    args["use_end"] = use_end
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    trainer, train_loader = pipeline(args, save_dir, model_path)
    modes_discovery(trainer, save_dir, num_trials = 50, num_interpolation = 4, plot_gif = plot_gif, num_itr = num_itr)