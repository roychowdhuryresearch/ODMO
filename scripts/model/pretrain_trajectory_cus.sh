nohup python ./src/customization/trajectory_customization.py uestc pretrained cuda:0 > logs/log_trajectory_customization_uestc &
nohup python ./src/customization/trajectory_customization.py mocap pretrained cuda:1 > logs/log_trajectory_customization_mocap &
nohup python ./src/customization/trajectory_customization.py HumanAct12 pretrained cuda:2 > logs/log_trajectory_customization_HumanAct12 &