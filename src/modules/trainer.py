import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import os
import wandb
import time
import copy

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from plotly.subplots import make_subplots
from draw_func.draw_gif import visualize_motion, plot_3d_motion_v2
import param.data_params as data_params

from modules.utils import plot_embeddings, clean_folder, outer_product, tf_optimizer
from modules.initializer import initialize_dataset_spec, initialize_dataset_args_classnames
from modules.models import ContrastiveLoss 
from modules.meters import *

import time

class Trainer:
    def __init__(
        self,
        model,
        evaluator,
        optimizers,
        data_loaders,
        args,
        save_dir,
        device,
        dataset_name
    ):
        super().__init__()
        self.model = model
        self.evaluator = evaluator
        self.optimizerG = optimizers["generator"]
        self.CEL = nn.CrossEntropyLoss()
        self.BCE = nn.BCELoss()
        self.L2Loss = nn.MSELoss(reduction="none")
        self.L1Loss = nn.L1Loss(reduction="none")
        self.ContrastiveLoss = ContrastiveLoss(margin=args["Contrastive_m"])
        self.train_loader = data_loaders["train"]
        self.valid_loader = data_loaders["valid"]
        self.args = args
        self.device = device
        self.max_len = args["motion_length"]
        if "tf_max" in args:
            tf_max = args["tf_max"]
        else:
            tf_max = 1
        self.tf_opt = tf_optimizer(
            total_epoch=args["tf_epochs"], mode=args["tf_mode"], tf_min=args["tf_min"], tf_max=tf_max)
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.y_weights = self.args["y_weights"]
        self.loss_type = self.args["loss_type"]
        if self.loss_type == "L2":
            self.criterion = nn.MSELoss(reduction="none")
        elif self.loss_type == "L1":
            self.criterion = nn.L1Loss(reduction="none")
        self.validation_frequency = self.args["validation_frequency"]
        self.scale = 1
        self.sigma = self.args["sigma"]
        
        # ablation study: 
        if "use_traj_loss" in self.args:
            self.use_traj_loss = self.args["use_traj_loss"] # added
        else:
            self.use_traj_loss = True
        self.use_contrastive = self.args["use_contrastive"]
        self.model.use_lmp = self.args["use_lmp"]

    def tG_loss(self, loc_pred, loc_label):
        loss_x = self.criterion(loc_pred, loc_label)  # batch_size, 60, 3
        loss_x[:, :, 1] = self.y_weights * loss_x[:, :, 1]
        return torch.mean(loss_x, -1)

    def mG_loss(self, x, y):
        mse_loss = self.criterion(x, y)
        mse_loss[:, :, 1::3] = self.y_weights*mse_loss[:, :, 1::3]  # batch * frames * (joints * 3)
        mse_loss = torch.mean(mse_loss, -1)  # * mask
        return mse_loss  # + loss_style  # batch, frame_num

    def contrastive_loss(self, x1, y1, x2, y2):
        target = torch.sum(torch.logical_and(y1, y2), -1)
        return self.ContrastiveLoss(x1, x2, target)
    
    def generator_loss(self, traj, pred_traj, data, pred_motion, length):
        trajectory_loss = self.tG_loss(pred_traj, traj)
        motion_loss = self.mG_loss(pred_motion, data)
        consistent_loss = torch.mean(self.criterion(pred_motion[:, :, :3], pred_traj), -1)
        trajectory_loss += consistent_loss
        return (
            torch.sum(trajectory_loss, -1),
            torch.sum(motion_loss, -1),
        )  # batchsize,  1
       
    def kld_loss(self, x_dist, mu):
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(mu)*self.sigma)
        kl_div = kl_divergence(x_dist, p_z).sum(1)
        return kl_div

    def one_batch(self, s1, s2, mtf=0):
        data_1, c_onehot_1, length_1 = (
            s1[0].to(self.device).float(),
            s1[1].to(self.device).float(),
            s1[2].to(self.device).float(),
        )
        data_2, c_onehot_2, length_2 = (
            s2[0].to(self.device).float(),
            s2[1].to(self.device).float(),
            s2[2].to(self.device).float(),
        )

        traj_1, pred_traj_1, dist_1, mu_1, pred_motion_1 = self.model(
            data_1, c_onehot_1, length_1, mtf=mtf)
        traj_2, pred_traj_2, dist_2, mu_2, pred_motion_2 = self.model(
            data_2, c_onehot_2, length_2, mtf=mtf)
        
        generator_tloss_1, generator_mloss_1 = self.generator_loss(
            traj_1, pred_traj_1, data_1, pred_motion_1, length_1
        )
        generator_tloss_2, generator_mloss_2 = self.generator_loss(
            traj_2, pred_traj_2, data_2, pred_motion_2, length_2
        )
        contrastive_loss = self.contrastive_loss(mu_1, c_onehot_1, mu_2, c_onehot_2)

        kld_loss = self.kld_loss(dist_1, mu_1) + self.kld_loss(dist_2, mu_2)

        return (
            generator_tloss_1,
            generator_mloss_1,
            generator_tloss_2,
            generator_mloss_2,
            contrastive_loss,
            kld_loss
        )

    def validation(self, dataloader, image_folder=None):
        self.model.enable_eval()
        meter = ValidMeter(dataset_name=self.dataset_name)
        with torch.no_grad():
            for _, (s1, _) in enumerate(dataloader, 0):
                data_1, c_onehot_1, length_1 = (
                    s1[0].to(self.device).float(),
                    s1[1].to(self.device).float(),
                    s1[2].to(self.device).float(),
                )
                traj_1, pred_traj_1, dist_1, mu_1, pred_motion_1 = self.model(
                    data_1, c_onehot_1, length_1
                )
                meter.add(pred_motion_1, c_onehot_1, mu_1)
        fig = make_subplots(rows=1, cols=1)
        _, label = initialize_dataset_args_classnames(self.dataset_name)
        stats = self.validation_sample(self.train_loader, repeat_num=200)
        z_text = [[str(y) for y in x] for x in stats["Confusion"]]
        fig2 = ff.create_annotated_heatmap(
            stats["Confusion"], x=label, y=label, annotation_text=z_text, colorscale='blues')
        stats.update({"confusion_fig": fig2})
        del stats["Confusion"]
        del stats["Mode"]
        stats.update({"emb_fig": fig})
        stats.update(
            {
                "generated_fig": meter.plot_traj(pred_traj_1.cpu().numpy()),
                "real_traj": meter.plot_traj(traj_1.cpu().numpy()),
            }
        )
        return stats

    def validation_sample(self, dataloader, repeat_num=15):
        self.model.enable_eval()
        num_class = self.model.tgenerator.class_dim
        (s_1, _) = next(iter(dataloader))  # dummpy
        ground_truth = s_1[0][:repeat_num].float().to(self.device)

        GMMs, mapping, gt_code_list, gt_end_list = self.model.create_GMM(
            dataloader, num_class
        )
        prediction_list, code_list, c_onehot_list = self.model.sample_motion(
            ground_truth,
            GMMs, mapping, gt_code_list, gt_end_list,
            num_samples=repeat_num,
        )
        c_onehot = torch.cat(c_onehot_list, 0)
        c_class_num = torch.argmax(c_onehot, -1)
        prediction = torch.cat(prediction_list, 0)
        stats = self.evaluator.evaluate_metrics(prediction, c_class_num)
        return stats

    def predict_npz_samples(self, dataloader, sampling_strategy, repeat_num=15, num_itr=10, dump_datadir=""):
        num_class = self.model.tgenerator.class_dim
        output_dim = self.model.mgenerator.output_dim
        self.model.enable_eval()
        clean_folder(dump_datadir)
        elapsed_times = []
        for i in range(10):
            torch.manual_seed(i)
            np.random.seed(i)
            random.seed(i)
            (s_1, _) = next(iter(dataloader))
            ground_truth = s_1[0][:repeat_num].float().to(self.device)
            GMMs, mapping, gt_code_list, gt_end_list = self.model.create_GMM(
                dataloader, num_class, num_itr=num_itr
            )
            st = time.time()
            prediction_list, code_list, c_onehot_list = self.model.sample_motion(
                ground_truth,
                GMMs, mapping, gt_code_list, gt_end_list,
                num_samples=repeat_num,
                sample_strategy=sampling_strategy
                # motionft_dim=output_dim,
            )
            et = time.time()
            elapsed_time = et - st
            elapsed_times.append(elapsed_time)
            data_arr = np.concatenate([item.cpu().numpy()
                                      for item in prediction_list])
            label_num = np.concatenate(
                [np.argmax(item.cpu().numpy(), -1) for item in c_onehot_list])
            np.savez(os.path.join(
                dump_datadir, f"GMM_sample_{repeat_num}_{i}"), motion=data_arr, label=label_num)
        #print(elapsed_times)

    def train(self, epochs):
        best_acc = 0 
        for e in tqdm(range(epochs)):
            self.model.enable_train()
            meter = Meter()
            tf_rate = self.tf_opt.step()
            for mb, (s1, s2) in enumerate(self.train_loader, 0):
                self.optimizerG.zero_grad()
                gtloss_1, gmloss_1, gtloss_2, gmloss_2, ctr_loss, kld_loss = self.one_batch(s1, s2, tf_rate)
                traj_loss = torch.mean(gtloss_1 + gtloss_2)
                motion_loss = torch.mean(gmloss_1 + gmloss_2)
                ctr_loss = torch.mean(ctr_loss)
                kld_loss = torch.mean(kld_loss)
                loss = motion_loss + kld_loss
                if self.use_traj_loss:
                    loss += traj_loss
                if self.use_contrastive:
                    loss += ctr_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizerG.step()
                meter.add(traj_loss.item(), motion_loss.item(),
                          ctr_loss.item(), loss.item(), kld_loss.item())
            log_info = meter.dump_wandb()

            if e % self.validation_frequency == 0 and e != 0:
                stats = self.validation(self.valid_loader)
                log_info.update(stats)

            wandb.log(log_info)
            if e % self.validation_frequency == 0 and e != 0:
                torch.save(
                    self.model.state_dict(), os.path.join(
                        self.save_dir, f"E_{e}.pkg")
                )
                if best_acc < log_info["Accuracy"] and e > 5000:
                    best_acc = log_info["Accuracy"]
                    torch.save(
                        self.model.state_dict(), os.path.join(
                        self.save_dir, f"best_{e}.pkg")
                    )
