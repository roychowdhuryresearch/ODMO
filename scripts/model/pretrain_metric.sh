nohup python ./src/metrics/calculate_metric.py mocap_end_release_MM > logs/log_metric_pretrain_mocap_end_MM &
nohup python ./src/metrics/calculate_metric.py HumanAct12_end_release_MM > logs/log_metric_pretrain_HumanAct12_end_MM &
nohup python ./src/metrics/calculate_metric.py uestc_end_release_MM train > logs/log_metric_pretrain_uestc_end_MM_train &
nohup python ./src/metrics/calculate_metric.py uestc_end_release_MM test > logs/log_metric_pretrain_uestc_end_MM_test &