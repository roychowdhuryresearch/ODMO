import torch
import torch.nn as nn
import torch.optim as optim
from random import random
import random
import numpy as np
import time
from modules.utils import outer_product
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MotionGenerator(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        n_layers,
        dropout,
        output_len,
        device,
    ):
        super().__init__()
        ## encoder
        self.encoder_embedding = nn.Linear(3, hid_dim)
        self.encoder_rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True) 
        
        ## decoder
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True)
        self.device = device
    
        self.embedding = nn.Linear(input_dim, hid_dim) ## 75: 60+1+7*2
        self.relu = nn.PReLU()
        self.output_dim = output_dim
        self.out_fc = nn.Linear(hid_dim, output_dim)
        self.output_len = output_len
        self.tanh = nn.Tanh()
        
    def embed(self, src):
        batch = self.relu(self.embedding(src))
        return batch

    def project(self, batch):
        batch = self.out_fc(batch)
        return batch

    def extract_localmov_traj(self, src):
        feature_len = src.shape[2]
        src1 = (src - src[:, :, :3].repeat(1, 1, int(feature_len / 3))).clone()
        traj = src[:, :, :3].clone()
        return src1, traj

    def get_conditon_feature(self, label, class_embedding, batch_size):
        condition_feature = outer_product(label, class_embedding).reshape(
            batch_size, -1
        )
        #condition_feature = condition_feature.detach() ## why detach???
        return condition_feature

   
    def forward(self, src, label,class_embedding, traj, tf_rate=0, gt_traj=None, perturbation=None, fix_bonelen=False):
        batch_size = label.shape[0]
        feature_len = src.shape[2]
        if perturbation is not None:
            class_embedding += perturbation
        encoder_input = self.relu(self.encoder_embedding(traj))
        oo, traj_encoder_h = self.encoder_rnn(encoder_input)
        
        outputs = torch.zeros(
            self.output_len, batch_size, feature_len, requires_grad=True
        ).to(self.device)

        condition_feature = self.get_conditon_feature(label, class_embedding, batch_size)

        time_step = torch.zeros((batch_size, 1)).to(self.device).detach()
        hidden = traj_encoder_h
        decoder_input = torch.zeros((batch_size, feature_len), requires_grad=True).to(self.device)

        for t in range(self.output_len):
            time_step += 1.0 / self.output_len 
            
            generator_input = torch.cat(
                (decoder_input, condition_feature, time_step), -1)
            embedding = self.embed(generator_input)
            oo, hidden = self.rnn(embedding.unsqueeze(1), hidden)
            output = self.project(oo)
            output = output.squeeze()
            teacher_force = True if random.random() < tf_rate else False
            if teacher_force:
                decoder_input = src[:, t, :]
            else:
                decoder_input = output

            outputs[t] = output

        outputs = outputs.permute(1, 0, 2)

        return outputs


class LMPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hid_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_hid_layers, batch_first=True
        )
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, output_dim*2),
        )
    
    def split(self,x):
        return x[:, :self.output_dim], x[:, self.output_dim:]
    
    def encode(self, x1):
        _, encode_hidden1 = self.encoder(x1.float())
        batch = self.fc(encode_hidden1[0][-1, :, :].float())
        mu, logvar = self.split(batch)
        #q_z_x = Normal(mu, logvar.mul(.5).exp())
        q_z_x = Normal(mu-mu, logvar.mul(.5).exp())
        return mu, q_z_x

class TrajGenerator(nn.Module):
    def __init__(self, code_dim, hidden_dim, n_layers, trajectory_dim, motion_length, num_class, use_end ,device):
        super().__init__()

        self.motion_len = motion_length
        self.hidden_dim = hidden_dim
        self.trajectory_dim = trajectory_dim
        self.class_dim = num_class
        input_dim = self.class_dim * code_dim + trajectory_dim
        self.expand = nn.Linear(input_dim, input_dim * motion_length)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.out_fc0 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, trajectory_dim)
        self.activation_fun = nn.PReLU()
        self.device = device
        self.tanh = nn.Tanh()
        self.use_end = use_end
    
    def get_conditon_feature(self, label, class_embedding, batch_size):
        condition_feature = outer_product(label, class_embedding).reshape(
            batch_size, -1
        )
        return condition_feature
    def forward(self, batch, label, end_point):
        batch_size = batch.shape[0]
        code_condition = self.get_conditon_feature(label, batch, batch_size)
        if not self.use_end:
            end_point = end_point * 0
        batch = torch.cat((code_condition, end_point), -1)   ## batch_size, 7*5 + 3
        batch = self.activation_fun(self.expand(batch))
        batch = batch.reshape(batch_size, self.motion_len, -1) ## batch_size, 38-> batch_size, 38*60
        batch = self.embedding(batch)
        oo, h = self.rnn(batch) ## batch_size, time, 128
        output = self.activation_fun(self.out_fc0(oo.squeeze())) ##
        
        output = self.out_fc1(output) ## batch_size, time, hidden_dim
        output = self.activation_fun(output)
        output = self.activation_fun(self.out_fc2(output))
        output = self.out_fc(output)
        #output = self.tanh(self.out_fc(output))
        return output
    
    def to_delta(self, x):
        return x[:,1:] - x[:,0:-1]
    def to_location(self, x):
        zeros = torch.zeros((x.shape[0], 1, x.shape[-1])).to(self.device)
        xx = torch.cat((zeros, x), 1)
        return torch.cumsum(xx, 1)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin = 5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if self.margin > 2:
            self.inclass_margin_ratio = 0.1
        else:
            self.inclass_margin_ratio = 0.5 ## 0.5 for dense margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            #target.float() * F.relu(self.margin*self.inclass_margin_ratio - (distances + self.eps).sqrt()).pow(2)
            target.float() * F.relu(-0.25 + distances) +
            (1 + -1 * target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        
        """
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        """
        return losses