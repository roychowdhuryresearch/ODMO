import torch
import torch.nn as nn
from scipy import stats
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics.cluster import homogeneity_score
from sklearn import metrics
class Ensemble(nn.Module):
    def __init__(self, generator, LMPencoder, mgenerator, device):
        super().__init__()
        self.tgenerator = generator
        self.LMPencoder = LMPencoder
        self.mgenerator = mgenerator
        self.device = device
        self.use_lmp = True

    def enable_train(self):
        self.tgenerator.train()
        self.LMPencoder.train()
        self.mgenerator.train()

    def enable_eval(self):
        self.tgenerator.eval()
        self.LMPencoder.eval()
        self.mgenerator.eval()

    def extract_localmov_traj(self, src):
        feature_len = src.shape[2]
        #src = src.clone()
        src1 = (src - src[:, :, :3].repeat(1, 1,int(feature_len / 3))).clone().detach()
        traj = src[:, :, :3].clone().detach()
        return src1, traj

    def get_embedding(self, src):
        src1, traj = self.extract_localmov_traj(src)
        if self.use_lmp:
            mu, dist = self.LMPencoder.encode(src1)
        else:
            mu, dist = self.LMPencoder.encode(src)
        return mu, dist, traj
    
    def forward(self, data, c_onehot, length, mtf=0):
        mu, dist, traj = self.get_embedding(data)
        code = dist.rsample() + mu
        pred_velocity = self.tgenerator(code, c_onehot, traj[:, -1, :])
        pred_traj = self.tgenerator.to_location(pred_velocity)
        pred_motion = self.mgenerator(
            data, c_onehot, code, pred_traj, tf_rate=mtf, gt_traj=traj, fix_bonelen=False)
        return traj, pred_traj, dist, mu, pred_motion
    
    def reconstruct(self, data, c_onehot, sigma_ratio = 1 ,mtf=0):
        mu, dist, traj = self.get_embedding(data)
        code = dist.rsample() * sigma_ratio + mu
        pred_velocity = self.tgenerator(code, c_onehot, traj[:, -1, :])
        pred_traj = self.tgenerator.to_location(pred_velocity)
        pred_motion = self.mgenerator(
            data, c_onehot, code, pred_traj, tf_rate=mtf, gt_traj=traj, fix_bonelen=False)
        return traj, pred_traj, dist, mu, pred_motion
    
    def create_GMM(self, dataloader, num_class, num_itr = 5):
        GMM_data_code, GMM_data_len, GMM_data_end = [], [], []
        for i in range(num_class):
            GMM_data_code.append([])
            GMM_data_len.append([])
            GMM_data_end.append([])
        code_l, class_num_l = [], []
        with torch.no_grad():
            for i in range(num_itr):
                for mb, (s1, s2) in enumerate(dataloader, 0):
                    data_1, c_onehot_1 = (
                        s1[0].to(self.device).float(),
                        s1[1].to(self.device).float(),
                    )
                    data_2, c_onehot_2 = (
                        s2[0].to(self.device).float(),
                        s2[1].to(self.device).float(),
                    )
                    #data_1 = torch.cat((data_1, data_2), dim = 0)
                    #c_onehot_1 = torch.cat((c_onehot_1, c_onehot_2), dim = 0)
                    code, dist, traj = self.get_embedding(data_1)
                    code = code.cpu().numpy()
                    code_l.append(code)
                    class_num = torch.argmax(c_onehot_1, -1).cpu().numpy()
                    class_num_l.append(class_num)
                    for c_idx, c in enumerate(code):
                        GMM_data_code[class_num[c_idx]].append(
                            c.reshape(1, -1))
                        GMM_data_end[class_num[c_idx]].append(traj[c_idx].cpu().numpy()[-1:, :3].reshape(-1, 3))
        
        overall_code, class_num_l = np.concatenate(code_l), np.concatenate(class_num_l)
        unique_c, unique_index = np.unique(np.concatenate(code_l), axis= 0,return_index=True)
        class_num_l = class_num_l[unique_index]
        overall_code = unique_c
        #print(overall_code[:3])
        gmm = GaussianMixture(n_components=num_class, random_state=0, warm_start= True, init_params="kmeans", tol=0.0001, max_iter=200)
        pred_clusterids = gmm.fit_predict(overall_code)
        filterd_code = []
        filtered_class_num_l = []
        for i in range(num_class):
            data_inds = np.where(class_num_l == i)[0]
            gmm_pre = pred_clusterids[data_inds]
            predicted_label = stats.mode(gmm_pre)[0][0] 
            select_index = np.where(gmm_pre == predicted_label)[0]
            filterd_code.append(overall_code[data_inds][select_index])
            filtered_class_num_l.append(class_num_l[data_inds][select_index])
        
        filtered_class_num_l = np.concatenate(filtered_class_num_l) 
        gmm.fit(np.concatenate(filterd_code))
        gmm.fit(np.concatenate(filterd_code))
        pred_clusterids = gmm.predict(np.concatenate(filterd_code))
        cluster_class_mapping = {}
        for i in range(num_class):
            data_inds = np.where(filtered_class_num_l == i)[0]
            gmm_pre = pred_clusterids[data_inds]
            predicted_label = stats.mode(gmm_pre)[0][0]
            cluster_class_mapping[i] = predicted_label
        #print(homogeneity_score(pred_clusterids, filtered_class_num_l))
        code_list, end_list = [], []
        for idx, GMM_d in enumerate(GMM_data_code):
            d = np.concatenate(GMM_d, 0)
            end = np.concatenate(GMM_data_end[idx], 0)
            code_list.append(d)
            end_list.append(end)

        return gmm, cluster_class_mapping, code_list, end_list

    def find_closest_end(self, code, gt_code, gt_end):
        dist = torch.cdist(gt_code.float(), code.float(), p=2)
        closest_pt_inds = torch.argmin(dist, 0)
        end_selected = gt_end[closest_pt_inds]
        return end_selected

    def sample_code_multimodal(self, gt_code_list, num_samples):
        code_list = []
        class_code = gt_code_list 
        class_code = np.unique(class_code, axis = 0)
        s_pre = -1
        K = np.arange(3,10)
        for i, k in enumerate(K):
            model = GaussianMixture(n_components=k, n_init=5, init_params='kmeans', random_state=0)
            labels = model.fit_predict(class_code)
            s = metrics.silhouette_score(class_code, labels, metric='euclidean')
            if s < s_pre * 0.95:
                break
            else:
                num_multimodality = k 
            s_pre = s 
        gm = GaussianMixture(n_components=num_multimodality, n_init=5, init_params='kmeans', random_state=0).fit(class_code)
        prediction = gm.predict(class_code)
        _, num_candidate_samples = np.unique(prediction, return_counts=True)
        th = 10
        noisy_cluster_index = np.where(num_candidate_samples <= th)[0]
        good_cluster_index = np.where(num_candidate_samples > th)[0]
        num_candidate_samples[noisy_cluster_index] = 0
        num_candidate_samples = (num_candidate_samples/sum(num_candidate_samples) * num_samples).astype(int) # added 
        num_candidate_samples[good_cluster_index[0]] += num_samples - sum(num_candidate_samples)
        mode_index = []
        for i in range(num_multimodality):
            code_list.append(np.random.multivariate_normal(
                gm.means_[i], gm.covariances_[i], num_candidate_samples[i]))
            mode_index.append(np.zeros(num_candidate_samples[i]) + i)
        return np.concatenate(code_list), np.concatenate(mode_index) 

    def sample_motion(self, data, GMMs, mapping, gt_code_list, gt_end_list, num_samples=5, sample_strategy="standard"):
        # data : dummy data only for feature size
        GMM_data_end = []
        for i in range(len(mapping)):
            GMM_data_end.append(GaussianMixture(n_components=1, random_state=0).fit(gt_end_list[i]))

        pred_motion_list, code_list, c_onehot_list = [], [], []
        #print("using sample strategy ", sample_strategy)
        with torch.no_grad():
            # class iter
            for i in range(len(mapping)):
                if sample_strategy=="standard":
                    sampled_code = np.random.multivariate_normal(
                            GMMs.means_[mapping[i]], GMMs.covariances_[mapping[i]], num_samples)
                elif sample_strategy=="MM":
                    sampled_code, _ = self.sample_code_multimodal(gt_code_list[i], num_samples)
                else:
                    raise NotImplementedError(f'{sample_strategy} is not implemented')
                    
                code = torch.from_numpy(sampled_code)
                end = self.find_closest_end(code, torch.from_numpy(
                    gt_code_list[i]), torch.from_numpy(gt_end_list[i]))
                c_onehot = torch.from_numpy(
                    np.zeros(len(mapping))).repeat(num_samples, 1)
                c_onehot[:, i] = 1
                code, c_onehot, end = (
                    code.to(self.device).float(),
                    c_onehot.to(self.device).float(),
                    end.to(self.device).float()
                )
                pred_traj = self.tgenerator(code, c_onehot, end)
                pred_traj = self.tgenerator.to_location(pred_traj)
                pred_motion = self.mgenerator(
                    data, c_onehot, code, pred_traj, tf_rate=0, gt_traj=pred_traj, fix_bonelen=True
                )
                pred_motion = pred_motion
                pred_motion_list.append(pred_motion)
                code_list.append(code)
                c_onehot_list.append(c_onehot)
        return pred_motion_list, code_list, c_onehot_list
