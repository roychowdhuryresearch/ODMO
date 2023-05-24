# ODMO
[ACMMM 2022] Official PyTorch Implementation of ["Action-conditioned On-demand Motion Generation". ACM MultiMedia 2022](https://dl.acm.org/doi/10.1145/3503161.3548287).

This repo contains the official implementation of our paper:

Action-conditioned On-demand Motion Generation
Qiujing Lu*, Yipeng Zhang*, Mingjian Lu, Vwani Roychowdhury
ACMMM 2022

### [**Project page**](https://roychowdhuryresearch.github.io/ODMO_ACMMM2022/) 

### Bibtex 
If you find our project is useful in your research, please cite:
```
@inproceedings{lu2022action,
  title={Action-conditioned On-demand Motion Generation},
  author={Lu, Qiujing and Zhang, Yipeng and Lu, Mingjian and Roychowdhury, Vwani},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

# Installation 
### Dependencies
Anaconda is recommended to create the virtual environment
```
conda env create -f environment.yml
conda activate ODMO
```

### For those who use pip (that may not work for 4090)
```
pip install -r requirements.txt
```

### Clean up the directory

```
sh ./scripts/cleanup.sh
```

### Datasets
```
sh ./scripts/download/download_dataset.sh
```
### Pretrained Models (including motion classifiers)
```
sh ./scripts/download/download_pretrain.sh
```
### If download (gdown) failed, please download from the link in the scripts and unzip it in to the home directory


# Training and Evaluation 
## **For generating metric based on the pretrained model**
1. sample real motion (this may take a while, for about 5 minutes)
```
sh ./scripts/model/sample_realdata.sh
```
Please go to the ./logs/ folder to check the status and wait for *"Sample real data from {dataset_name} is accomplished"*

2. sample the pretrained model (please be aware of the cuda device in the script)
```
sh ./scripts/model/pretrain_inference.sh
```
Please go to the ./logs/ folder to check the status and wait for *"Sample data from {model_name} is accomplished"*

3. generating metric based on classifier
```
sh ./scripts/model/pretrain_metric.sh
```
Please see *logs/log\_{task}\_{dataset\_name}\_{model\_name}* for numbers

4. modes discovery 
```
sh ./scripts/model/pretrain_modes_discovery.sh
```
5. trajectory customization 
```
sh ./scripts/model/pretrain_trajectory_cus.sh
```
The dist_e for 10 different seeds can be found in the csv under *./results/endloc_customization/* 

## Train your network on three datasets
### **(Caution!)** it will take long time if you use default parameters
### Please review the hyperparameter in ./src/param/param_{dataset_name}.py
### It is better to use wandb to track it, or you can add the following line in the beginning of the main function
```
os.environ['WANDB_MODE'] = 'offline'
```

### MoCap
```
sh ./scripts/model/train_odmo_mocap.sh
```
### HumanAct12
```
sh ./scripts/model/train_odmo_humanact12.sh
```
### UESTC
```
sh ./scripts/model/train_odmo_uestc.sh
```
## Note 
Each time of training, it will generate a folder with unique name (we call it model name) under the ckpt/{dataset_name} folder, please keep tracking the most recent ones or you can use wandb to track it 

## Inference the model 
```
python inference_odmo_best.py {model_name} {sampling strategy} {device}
```
For example we want to inference *mocap_abc* using *mode_preserve_sampling* on cuda:0

```
python ./src/inference_odmo_best.py mocap_abc MM cuda:0
```

or by using the conventional sampling

```
python ./src/inference_odmo_best.py mocap_abc standard cuda:0
```

## Metric based on Classifier
It's a cpu task. we need the folder name of the sampled data in ./sampled_data folder

```
python ./src/metrics/calculate_metric.py mocap_end_release_MM
```
If we want to calculate metric on UESTC dataset (it has both train/test set)
we can use
```
python ./src/metrics/calculate_metric.py uestc_end_release_MM train
python ./src/metrics/calculate_metric.py uestc_end_release_MM test
```

You also can use 
```
./src/metrics/calculate_metric_ext.py
```
to calculate metric from any specific sampled data

## Metric on APD

This is also a CPU task
```
python src/metrics/calculate_apd.py mocap_end_release_MM
```
Similarly, we can use the following command for uestc dataset
```
python ./src/metrics/calculate_apd.py uestc_end_release_MM train
python ./src/metrics/calculate_apd.py uestc_end_release_MM test
```

## Modes interpolation
The modes_discovery_interpolation.py can handle both pretrained model and the customized trained model. 

For pretrained model
```
python ./src/customization/modes_discovery_interpolation.py {dataset_name} pretrained {use_end} {device}
```

For customized trained model, the model name is the name randomly generated in your ckpt/{dataset} folder
```
python ./src/customization/modes_discovery_interpolation.py {dataset_name} {model_name} {use_end} {device}
```

## Trajectory customization and metrics
The trajectory_customization.py can handle both pretrained model and the customized trained model. 

For pretrained model
```
python ./src/customization/trajectory_customization.py {dataset_name} pretrained {device}
```

For customized trained model, the model name is the name randomly generated in your ckpt/{dataset} folder
```
python ./src/customization/trajectory_customization.py {dataset_name} {model_name} {device}
```
The dist_e for 10 different seeds can be found in the csv under *./results/endloc_customization/* 

## Plotting the sampled motion
If we want to plot the sampled motion from the inference*.py, we can use the 
```
./src/draw_func/draw_gif_from_np_multi.py
```
in that file you can specify the datasetname, npz file name, output folder, downsample_rate and n_jobs.


