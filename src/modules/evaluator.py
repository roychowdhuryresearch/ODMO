
from train_classifiers_a2m import AMotionDiscriminator, AMotionDiscriminatorForFID
import numpy as np
import os
import sys
import inspect
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
from scipy import linalg
from modules.utils import *
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader

def extract_localmov_traj(src):
    # given shape  is batch size, motion_len and ft_len
    feature_len = src.shape[2]
    src1 = src - np.tile(src[:, :, :3], (1, 1, int(feature_len / 3)))
    traj = src[:, :, :3]
    return src1, traj


def calculate_activation_statistics(activations):

    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity_multimodality0(activations, labels, num_labels):
    # print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    # print('=== Evaluating Multimodality ===')
    multimodality = 0
    labal_quotas = np.repeat(multimodality_times, num_labels)
    # !!!!!!Setting for removing wash class
    labal_quotas[1] = 0
    while np.any(labal_quotas > 0):
        # print(labal_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not labal_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        labal_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation, second_activation)

    multimodality /= multimodality_times * num_labels

    return diversity, multimodality


def calculate_diversity_multimodality(activations, labels, num_labels):
    # print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)
    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    # print('=== Evaluating Multimodality ===')
    multimodality = 0
    labal_quotas = np.repeat(multimodality_times, num_labels)
    while np.any(labal_quotas > 0):
        # print(labal_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not labal_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        labal_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation, second_activation)

    multimodality /= multimodality_times * num_labels

    return diversity, multimodality


class Meter():
    def __init__(self):
        self.prediction = []  # GPU and Torch
        self.label = []
        self.MSE_loss = []
        self.style_loss = []

    def add(self, prediction, label):
        self.prediction.append(prediction.detach())
        self.label.append(label.detach())

    def add_loss(self, mse_loss, style_loss):
        self.MSE_loss.append(mse_loss)
        self.style_loss.append(style_loss)

    def dump_prediction(self):
        if len(self.prediction) > 1:  # self.prediction is a torch array
            prediction = torch.cat(self.prediction, 0)
            label = torch.cat(self.label, 0)
        else:
            prediction = self.prediction[0]
            label = self.label[0]
        label = torch.argmax(label, dim=-1)
        # align our model label with AC trained classifier label
        ind = torch.nonzero(label)
        label[ind] += 1
        return prediction, label

    def get_loss(self):
        if len(self.MSE_loss) > 1:
            loss = torch.cat(self.MSE_loss, 0)
            style_loss = torch.cat(self.style_loss)
        else:
            loss = self.MSE_loss[0]
            style_loss = self.style_loss[0]
        return torch.mean(loss).item(), torch.mean(style_loss).item()


class Evaluator:
    def __init__(self, num_class, motion_len, c_input_size, c_hidden_dim, c_hidden_layer, c_path, dataset_name="mocap", classifier_mode="local_mv", param_dict=None, device="cuda:0", classifier_type="ours", option="train"):
        self.device = device
        self.dataset_name = dataset_name
        self.param_dict = param_dict
        self.num_class = num_class
        self.clf_hidden_layer = c_hidden_layer
        self.clf_input_size = c_input_size
        self.clf_hidden_dim = c_hidden_dim
        self.c_path = c_path
        self.motion_len = motion_len
        self.classifier_type = classifier_type
        self.classifier_mode = classifier_mode
        self.option = option
        self.clf, self.clf_fid = self._load_classifier()
        self.clf.eval()
        self.clf_fid.eval()
        self._load_gt_stats()
        self.entropy_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        

    def _load_classifier(self):
        classifier = AMotionDiscriminator(self.clf_input_size, self.clf_hidden_dim , self.clf_hidden_layer,
                                            self.num_class, device=self.device,  classifier_mode=self.classifier_mode).to(self.device)
        classifier_fid = AMotionDiscriminatorForFID(self.clf_input_size, self.clf_hidden_dim, self.clf_hidden_layer ,
                                                    self.num_class, device=self.device,  classifier_mode=self.classifier_mode).to(self.device)
        model = torch.load(self.c_path, map_location=self.device)
        if 'model' in model:
            model = model['model']
        classifier.load_state_dict(model)
        classifier.eval()
        classifier_fid.load_state_dict(model)
        classifier_fid.eval()
        return classifier, classifier_fid

    def to_device(self, x):
        return torch.from_numpy(x).to(self.device)

    def _load_gt_stats(self):
        if self.dataset_name == "mocap":
            fn = "./sampled_data/mocap/real/mocap_100_400_0.npz"
        elif self.dataset_name == "HumanAct12":
            fn = "./sampled_data/HumanAct12/real/HumanAct12_60_250_0.npz"
        if self.dataset_name == "uestc":
            if self.option == "test":
                fn = './sampled_data/uestc/real/test/uestc_test_60_200_0.npz'
            else:
                fn = "./sampled_data/uestc/real/train/uestc_60_200_0.npz"
        loaded = np.load(fn)
        gt_motion = torch.from_numpy(loaded["motion"])
        gt_label = torch.from_numpy(loaded["label"])
        gt_motion = gt_motion.to(self.device)
        gt_label = gt_label.to(self.device)
        self.gt_stats, activation = self.get_stats(gt_motion)
        self.gt_diversity, self.gt_multimodality = calculate_diversity_multimodality(
            activation, gt_label, self.num_class)

    def get_entropy_of_prediction(self, pred_prob, gt_labels):
        class_entropy = torch.mean(
            self.entropy_criterion(pred_prob, gt_labels), 0)
        return class_entropy

    def get_stats(self, motion):
        activations = self.calculate_activations_labels(motion)
        stats = calculate_activation_statistics(activations)
        return stats, activations

    def calculate_activations_labels(self, motion):
        classifier = self.clf_fid
        motion = torch.clone(motion).float().detach_()
        with torch.no_grad():
            pre = classifier(motion, None)
        return pre.detach()

    def cal_model_acc(self, motion, label, fn_name="tiger_o"):
        classifier = self.clf
        classifier.eval()
        #print("Calculating Accuracies in confusion matrix.....")
        confusion = torch.zeros(
            self.num_class, self.num_class, dtype=torch.long)
        with torch.no_grad():
            prob, _ = classifier(motion, None)
            pred = prob.detach().argmax(dim=-1)
        for label, pred in zip(label, pred):
            # print(label.data, pred.data)
            confusion[label][pred] += 1
        # print("confusion matrix is ", confusion)
        confusion_np = confusion.cpu().numpy()
        acc = np.sum(np.diag(confusion_np)) / np.sum(confusion_np)
        #print("acc:", acc)
        if fn_name is not None:
            plt.figure()
            ax = sns.heatmap(confusion_np, annot=True, fmt="d")
            plt.savefig(f"./evaluation/{fn_name}_confusion_mat.png")
        return confusion_np, acc, prob.detach()

    # compare across several generator
    def evaluate_metrics(self, motion, label, model_name="ground_truth"):
        pred_stats, pred_activation = self.get_stats(motion)
        pred_fid = self.calculate_fid(self.gt_stats, pred_stats)
        pred_diversity, pred_multimodality = calculate_diversity_multimodality(
            pred_activation, label, self.num_class)
        pred_cfm, pred_acc, pred_prob = self.cal_model_acc(motion, label, None)
        pred_entropy = self.get_entropy_of_prediction(pred_prob, label)
        res = {"Mode": model_name, "Accuracy": pred_acc, "FID": pred_fid, "Diversity": pred_diversity.item(),
               "Multimodality": pred_multimodality.item(), "Confusion": pred_cfm, "class_entropy": pred_entropy.item()}
        return res

    def calculate_fid(self, stats1, stats2):
        # extract features then compute FID
        mu1, sigma1 = stats1
        mu2, sigma2 = stats2
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
