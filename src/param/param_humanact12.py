import numpy as np 
import param.data_params as data_params
import os
param_dict={"dataset_name" : "HumanAct12",
# data path
"data_dir" : os.path.join(data_params.working_dir, "dataset/HumanAct12/HumanAct12"),
"csv_fn" : os.path.join(data_params.working_dir, "dataset/HumanAct12/humanact12.csv"),
# training parameter
"num_workers" : 1,
"lr" : 1e-3,
"batch_size" : 512,
"p_test" : 0.1,
"p_val" : 0.1,
"seed": 42,
######Models######
### motion generator
"motion_length" : 60 ,
"num_class" : 12,
"joint_num": 24,
"mg_num_hidlayers" : 2,
"mg_hidden_dim" : 128,
### embedding network size
"mu_dim" : 20,
"em_hidden_dim" : 64,
"em_num_hidlayers" : 1,
### trajectory network size
"tg_hidden_dim" : 128,
"tg_num_hidlayers":2, 
### use low body parts to predict trajectory velocity
"validation_frequency": 200,
"y_weights": 5,
"loss_type":"L2",
"device" : "cuda:0", 
"Contrastive_m": 5,
"sigma": 0.05,
"tf_epochs": 4000,
"tf_min": 0.3,
"tf_mode": "linear",
"use_weighted_sampling": True,
"use_end": True, 
### Classifer:
"classifier_mode" : "org",
"c_hidden_dim":128, 
"c_layers":2,
"use_lmp": True,
"use_contrastive": True,
"use_traj_loss": True,
"repeat_num":250,
"metric_classifier_path": "./ckpt/classifier/action_recognition_model_humanact12.tar"
}
param_dict["motionft_dim"]= param_dict["joint_num"] * 3
param_dict["mg_input_dim"] = param_dict["num_class"] * param_dict["mu_dim"] + param_dict["joint_num"] * 3 + 1