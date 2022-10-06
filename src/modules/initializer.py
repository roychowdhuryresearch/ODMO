from modules.evaluator import Evaluator
from modules.models import *
from modules.ensemble import *
from dataloader.dataloader_samplestart_rotate2xz_pairwise import (
    MotionDataset,
    MotionDatasetHumanAct12,
    MotionDatasetUestc,
    create_weighted_loaders
)
import param.data_params as data_params
import param.param_mocap as param_mocap
import param.param_humanact12 as param_humanact12
import param.param_uestc as param_uestc

def initialize_cuda(device):
    # Setup GPU optimization if CUDA is supported
    if torch.cuda.is_available():
        device = torch.device(device)
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        device = torch.device("cpu")
        print("CUDA NOT supported")
    return device


def initialize_dataset_args(dataset_name):
    if dataset_name == "mocap":
        args = param_mocap.param_dict
        args["lie_info"] = {"kinematic_tree": data_params.mocap_kinematic_chain,
                            "bone_offset": data_params.mocap_raw_offsets}
    elif dataset_name == "HumanAct12":
        args = param_humanact12.param_dict
        args["lie_info"] = {"kinematic_tree": data_params.humanact12_kinematic_chain,
                            "bone_offset": data_params.mocap_raw_offsets}
    elif dataset_name == "uestc":
        args = param_uestc.param_dict
        args["lie_info"] = {"kinematic_tree": data_params.uestc_kinematic_chain,
                            "bone_offset": data_params.mocap_raw_offsets}
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
    return args

def initialize_dataset_args_classnames(dataset_name):
    if dataset_name == "mocap":
        args = param_mocap.param_dict
        class_names = data_params.mocap_class_names
    elif dataset_name == "HumanAct12":
        args = param_humanact12.param_dict
        class_names = data_params.humanact12_class_names
    elif dataset_name == "uestc":
        args = param_uestc.param_dict
        class_names = data_params.uestc_class_names
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
    return args, class_names

def initialize_dataset_spec(dataset_name):
    if dataset_name == "mocap":
        class_names = data_params.mocap_action_enumerator
        num_joint = 20
        motionft_dim = num_joint * 3
        kinematic_tree = data_params.mocap_kinematic_chain
    elif dataset_name == "HumanAct12":
        class_names = data_params.humanact12_class_names
        num_joint = 24
        motionft_dim = num_joint * 3
        kinematic_tree = data_params.humanact12_kinematic_chain
    elif dataset_name == "uestc":
        class_names = data_params.uestc_class_names
        num_joint = 18
        motionft_dim = num_joint * 3
        kinematic_tree = data_params.uestc_kinematic_chain
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
    return class_names, num_joint, motionft_dim, kinematic_tree


def initialize_dataloaders(args):
    dataset_name = args["dataset_name"]
    num_workers = args["num_workers"]
    batch_size = args["batch_size"]
    p_val = args["p_val"]
    p_test = args["p_test"]
    use_weighted_sampling = args["use_weighted_sampling"]
    csv_fn = args["csv_fn"]
    data_dir = args["data_dir"]
    motion_length = args["motion_length"]
    dataset_name = args["dataset_name"]
    seed = args["seed"]
    if dataset_name == "HumanAct12":
        dataset = MotionDatasetHumanAct12(
            csv_fn, data_dir, max_length=motion_length
        )
    elif dataset_name == "mocap":
        dataset = MotionDataset(
            csv_fn, data_dir, max_length=motion_length
        )
    elif dataset_name == "uestc":
        dataset = MotionDatasetUestc(
            csv_fn, data_dir, max_length=motion_length
        )
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented')
        
    if args["device"] != "cpu":
        extras = {"num_workers": num_workers, "pin_memory": True}
    else:
        extras = False
    
    if use_weighted_sampling:
        train_loader, val_loader = create_weighted_loaders(
            dataset,
            batch_size=batch_size,
            p_val=p_val,
            shuffle=True,
            extras=extras,
            seed = seed
            )
    else:
        train_loader, val_loader, _ = create_kfold_loader(
            dataset,
            folder_num=1,
            batch_size=batch_size,
            p_val=p_val,
            p_test=p_test,
            extras=extras,
            seed = seed
        )
    
    dataloaders = {"train": train_loader,
                   "valid": val_loader}
    return dataloaders

def initialize_evaluator(args, classifier_type = "a2m"):
    c_hidden_layer = args["c_layers"]
    c_hidden_dim = args["c_hidden_dim"]
    c_path = args["metric_classifier_path"]
    device = args["device"]
    dataset_name = args["dataset_name"]
    num_class = args["num_class"]
    motion_length = args["motion_length"]
    motionft_dim = args["motionft_dim"]
    classifier_mode = args["classifier_mode"] 
    evaluator = Evaluator(num_class, motion_length, motionft_dim, 
                c_hidden_dim, c_hidden_layer, c_path, dataset_name=dataset_name, classifier_mode=classifier_mode,
                param_dict=args,device=device, classifier_type=classifier_type) 
    return evaluator

def initialize_model(args):
    num_class = args["num_class"]
    joint_num = args["joint_num"]

    motionft_dim = args["motionft_dim"]
    mg_input_dim = args["mg_input_dim"]
    mg_hidden_dim = args["mg_hidden_dim"]

    mg_num_hidlayers = args["mg_num_hidlayers"]
    mg_hidden_dim = args["mg_hidden_dim"]

    em_hidden_dim = args["em_hidden_dim"]
    em_num_hidlayers = args["em_num_hidlayers"]
    mu_dim = args["mu_dim"]
    tg_hidden_dim = args["tg_hidden_dim"]
    tg_num_hidlayers = args["tg_num_hidlayers"]
    device = args["device"]
    use_end = args["use_end"]
    motion_length = args["motion_length"]
    motionGenerator = MotionGenerator(
        input_dim=mg_input_dim,
        hid_dim=mg_hidden_dim,
        output_dim=motionft_dim,
        n_layers=mg_num_hidlayers,
        dropout=0.2,
        output_len=motion_length,
        device=device,
    )

    trajGenerator = TrajGenerator(
        code_dim=mu_dim,
        hidden_dim=tg_hidden_dim,
        n_layers=tg_num_hidlayers,
        trajectory_dim=3,
        motion_length=motion_length - 1,
        num_class=num_class,
        use_end=use_end,
        device=device,
    )

    emb_net = LMPEncoder(
        input_dim=motionft_dim,
        hidden_dim=em_hidden_dim,
        num_hid_layers=em_num_hidlayers,
        output_dim=mu_dim
    )

    model = Ensemble(trajGenerator.to(device), emb_net.to(
        device), motionGenerator.to(device), device)
    
    return model
