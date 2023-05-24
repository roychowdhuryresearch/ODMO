nohup python src/inference_odmo.py uestc MM use_end cuda:0 > logs/log_inference_uestc_MM &
nohup python src/inference_odmo.py mocap MM use_end cuda:1 > logs/log_inference_mocap_MM &
nohup python src/inference_odmo.py HumanAct12 MM use_end cuda:0 > logs/log_inference_HumanAct12_MM &
nohup python src/inference_odmo.py uestc MM no_end cuda:1 > logs/log_inference_uestc_MM_noend &
nohup python src/inference_odmo.py mocap MM no_end cuda:0 > logs/log_inference_mocap_MM_noend &
nohup python src/inference_odmo.py HumanAct12 MM no_end cuda:1 > logs/log_inference_HumanAct12_MM_noend &
