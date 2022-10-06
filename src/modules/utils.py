import torch
import torch.nn as nn
import torch.optim as optim
from random import random
import random
import numpy as np
import os
import shutil
from plotly import tools 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
import plotly.graph_objs as go
import math
from scipy import stats, linalg
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class tf_optimizer:
    def __init__(self, total_epoch=5000, mode="linear", tf_min=0, tf_max=1):
        self.total_epoch = total_epoch
        self.mode = mode
        self.tf_min = tf_min
        self.tf_max = tf_max
        self.cur_step = 0
        self.delta_tf = (tf_max - tf_min) / total_epoch
        self.exp_delta = np.exp(-np.arange(total_epoch)*(20.0/total_epoch))
        #self.exp_delta = np.exp(-np.arange(total_epoch)*0.02)

    def step(self):
        if self.mode == "linear":
            cur_tf = max(self.tf_max - self.cur_step * self.delta_tf, self.tf_min)
        elif self.mode == "linear_pump":
            #### two 3000. 3000
            #self.delta_tf = 1.0/3000
            module_cur_step = self.cur_step%10000
            cur_tf = max(1 - module_cur_step * self.delta_tf, self.tf_min)
        else:
            ### exp decay
            cur_tf = max(self.exp_delta[min(self.cur_step, self.total_epoch-1)], self.tf_min)

        self.cur_step += 1
        return cur_tf
        

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        os.mkdir(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)


def style_loss(output_pose, input_pose):
    criterion = nn.MSELoss(reduction="none")
    local_mv = output_pose - output_pose[:, :, :3].repeat(1, 1, 20)
    local_mv_y = input_pose - input_pose[:, :, :3].repeat(1, 1, 20)
    return criterion(local_mv, local_mv_y)


def outer_product(x, y):
    return torch.bmm(x.unsqueeze(2), y.unsqueeze(1))



def map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[ 0., 0., 0.],
                        [ 0., 0.,-1.],
                        [ 0., 1., 0.]])

    R_y = v.new_tensor([[ 0., 0., 1.],
                        [ 0., 0., 0.],
                        [-1., 0., 0.]])

    R_z = v.new_tensor([[ 0.,-1., 0.],
                        [ 1., 0., 0.],
                        [ 0., 0., 0.]])

    R = R_x * v[..., 0, None, None] + \
        R_y * v[..., 1, None, None] + \
        R_z * v[..., 2, None, None]
    return R

def map_to_lie_vector(X):
    """Map Lie algebra in ordinary (3, 3) matrix rep to vector.

    In literature known as 'vee' map.

    inverse of map_to_lie_algebra
    """
    return torch.stack((-X[..., 1, 2], X[..., 0, 2], -X[..., 0, 1]), -1)


def rodrigues(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map_to_lie_algebra(v / theta)

    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I + torch.sin(theta)[..., None]*K \
        + (1. - torch.cos(theta))[..., None]*(K@K)
    return R


def lie_exp_map(log_rot, eps: float = 1e-4):
    """
    Convert the lie algebra parameters to rotation matrices
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    R = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * torch.bmm(skews, skews)
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )
    # print(R.shape)
    return R


def hat(v):
    """
    compute the skew-symmetric matrices with a batch of 3d vectors.
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")
    h = v.new_zeros(N, 3, 3)
    x, y, z = v.unbind(1)
    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def create_rotation(data):
    ## ROTATION ALONG Y axis
    batch_size = data.shape[0]
    rotation = np.tile(np.diag([1, 1, 1]).astype("float"), (batch_size, 1, 1))
    sel_axis = [0, 2]
    vect_norm = np.linalg.norm(data[:, sel_axis], axis=1)
    xz_norm = data[:, sel_axis] / (vect_norm + 1e-8)[:, None]
    # print(xz_norm,rotation)
    rotation[:, 0, 0] = xz_norm[:, 0]
    rotation[:, 2, 2] = -xz_norm[:, 0]
    rotation[:, 0, 2] = xz_norm[:, 1]
    rotation[:, 2, 0] = xz_norm[:, 1]
    ## transpose matrix is the sam
    return rotation


def create_rotation_vect(vect):
    ## ROTATION ALONG Y axis
    rotation = np.diag([1, 1, 1]).astype("float")
    sel_axis = [0, 2]
    vect_norm = np.linalg.norm(vect[sel_axis])
    xz_norm = vect[sel_axis] / (vect_norm + 1e-8)
    # print(xz_norm,rotation)
    rotation[ 0, 0] = xz_norm[0]
    rotation[2, 2] = -xz_norm[ 0]
    rotation[ 0, 2] = xz_norm[1]
    rotation[2, 0] = xz_norm[ 1]
    ## transpose matrix is the sam
    return rotation


def draw_ellipse(mean, v, u ,i):
    #print("after",i,mean,v)
    angle = - np.arctan(u[1] / u[0])
    a, b = v[0], v[1]
    x_ = []
    y_ = []
    for t in range(0,361,10):
        x_tmp = a*(math.cos(math.radians(t))) 
        y_tmp = b*(math.sin(math.radians(t))) 
        x = math.cos(angle) * x_tmp - math.sin(angle) * y_tmp +  mean[0]
        y = math.sin(angle) * x_tmp + math.cos(angle) * y_tmp +  mean[1]           
        x_.append(x)    
        y_.append(y)
    elle = go.Scatter(x=x_ , y=y_, mode='lines',showlegend=False,line=dict(color=data_params.colors[i],width=2))
    return elle

def plot_embeddings(fig, embeddings, targets, class_info, xlim=None, ylim=None, row=1, col=1):
    # plt.figure(figsize=(10,10))
    
    max_class = len(class_info)

    gmm = GaussianMixture(n_components=max_class, random_state=0)
    pred_clusterids = gmm.fit_predict(embeddings)
    cluster_class_mapping = {}
    for i in range(max_class):
        data_inds = np.where(targets==i)[0]
        predicted_label = stats.mode(pred_clusterids[data_inds])[0][0]
        cluster_class_mapping[i] = predicted_label

    ft_len = embeddings.shape[1]
    if ft_len >2:
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
        
    for i in range(max_class):
        inds = np.where(targets == i)[0]
        trace_i = go.Scatter(x=embeddings[inds, 0], y=embeddings[inds, 1], marker_color=data_params.colors[i], mode="markers", name=f"Class {i} : {class_info[i]}")
        fig.add_trace(trace_i, row=row, col=col)
        mean = gmm.means_[cluster_class_mapping[i]]
        covar = gmm.covariances_[cluster_class_mapping[i]]
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        if ft_len > 2:
            mean = np.squeeze(pca.transform(mean.reshape(1,-1)))
            v = np.squeeze(pca.transform(v.reshape(1,-1))) 
            w = np.squeeze(pca.transform(w[0].reshape(1,-1)))
        u = w / linalg.norm(w)
        elle = draw_ellipse(mean, v , u ,i)
        fig.add_trace(elle, row=row, col=col)
  


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
