import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser
import random
import numpy as np
import torch
import yaml
from datetime import datetime
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
import json
from transformers import logging
logging.set_verbosity_error()

import data
import model
import probe
import regimen
import loss
import reporter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_new_experiment_dir(yaml_args):
    """Constructs a directory in which results and params will be stored.
    If yaml_args["results_dir"] is not None, then it is reused; no new directory is constrcted.
    
    Args:
      yaml_args: the global config dictionary loaded from yaml
    """

    if 'step' not in yaml_args['model']:
        now = datetime.now()
        date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]))
        model_suffix = '-'.join((yaml_args['model']['type'], yaml_args['dataset']['name']))
        new_root = os.path.join(yaml_args['reporting']['root'], date_suffix + '-' + model_suffix +'/' )

    else:
        if 'pythia' in yaml_args['model']['type']:
            new_root = os.path.join(yaml_args['reporting']['root'], "step-" + str(yaml_args['model']['step']))

    if yaml_args['verbose']:
        tqdm.write(f"Constructing new results directory at {new_root}\n\n")

    yaml_args['reporting']['root'] = new_root
    os.makedirs(new_root, exist_ok=True)

    try:
        yaml.safe_dump(yaml_args, open(os.path.join(yaml_args['reporting']['root'], os.path.basename(yaml_args["experiment_config"])), 'w'))

    except yaml.representer.RepresenterError:
        tqdm.write('Note, the config being used is the same as that already present in the results dir')


def choose_dataset_class(args):

    if args['model']['type'] == 'pythia':
        return data.PythiaDataset(args)

    elif args['model']['type'] == 'disk':
        return data.DiskDataset(args)
    
    else:
        raise ValueError('Invalid model type; choose (pythia, disk)')


def choose_model_class(args):

    if args['model']['type'] == 'pythia':
        return model.PythiaModel(args)
    
    elif args['model']['type'] == 'disk':
        return model.DiskModel(args)
    
    else:
        raise ValueError('Invalid model type; choose (pythia, disk)')


def choose_probe_class(args, layer):

    if "type" in args['probe'] and args['probe']['type'] == "dynamic":
        return probe.DynamicDimProbe(args, layer)
    
    return probe.FixedDimProbe(args, layer)

    

def execute_experiment(args):
    """
    Execute an experiment as determined by the configuration
    in args.
    """

    if args["gpu"]:
        args['device'] = "cuda:0"

    else:
        args['device'] = "cpu"

    expt_dataset = choose_dataset_class(args)
    args['model']['special_token_ids'] = expt_dataset.get_special_token_ids()
    args['dataset']['label_space_size'] = expt_dataset.get_num_labels()

    expt_model = choose_model_class(args)
    args['model']['hidden_dim'] = expt_model.get_hidden_dim()
    args['model']['num_layers'] = expt_model.get_num_layers()

    expt_loss = loss.CrossEntropyLoss()
    expt_reporter = reporter.SentenceReporter(args, expt_dataset)
    expt_regimen = regimen.ProbeRegimen(args, reporter=expt_reporter)


    # Split data into portions.

    def split_data_into_portions(dataset_train_dataset):
        total_len = len(dataset_train_dataset)
        fractions = list(map(float, args['regimen']['inds'].split(',')))
            
        train_portions = []
        eval_portions = []
        for i in range(len(fractions)):
            train_portions.append(torch.utils.data.Subset(dataset_train_dataset,
                                                          range(0, max(1, int(fractions[i] * total_len)))))
            if i != len(fractions) - 1:
                eval_portions.append(torch.utils.data.Subset(dataset_train_dataset,
                                                         range(max(1, int(fractions[i] * total_len)), max(2, int(fractions[i + 1] * total_len)))))

        return fractions, train_portions, eval_portions


    fractions, train_portions, eval_portions = split_data_into_portions(expt_dataset.train)


    dev_dataloader = expt_dataset.get_dev_dataloader()
    test_dataloader = expt_dataset.get_test_dataloader()

    # All layers + embedding layer
    if 'layer' in args['model'] and args['model']['layer'] == 'all': 
        layers = range(args['model']['num_layers'] + 1)

    # Only choosen layers
    elif 'layer' in args['model']:
        
        if "," in args['model']['layer']:
            layers = list(map(int, args['model']['layer'].split(',')))
        
        else:
            layers = [int(args['model']['layer'])]
    
    # defualt: Last layer 
    else: 
        layers = [int(args['model']['num_layers'])]

    # Probing over the portions.

    online_coding_list = {l: [] for l in layers}
    test_report_list = {l: [] for l in layers}
            
    for i in range(len(train_portions)):
      
        current_train = DataLoader(train_portions[i],
                                   batch_size=expt_dataset.batch_size,
                                   shuffle=False)

        # portions
        if i < len(eval_portions):
            current_dev = DataLoader(eval_portions[i],
                                    batch_size=expt_dataset.batch_size,
                                    shuffle=False)

        if args['verbose']:
            tqdm.write(f"\n\nTraining probe portion {fractions[i] * 100}% on layers {list(layers)}\n")


        expt_probe = {}
        for l in layers:
            set_seed(args['seed'])
            expt_probe[l] = choose_probe_class(args, layer=l)

        if i != len(train_portions)-1:
            evals = expt_regimen.train_until_convergence(expt_probe, expt_model, expt_loss,
                                                         current_train,
                                                         dev_dataloader,
                                                         portion=fractions[i],
                                                         layers=list(layers).copy(),
                                                         eval_datasets = {'dev': dev_dataloader,
                                                                          'test': test_dataloader,
                                                                          'online_portion': current_dev})
        else: 
            evals = expt_regimen.train_until_convergence(expt_probe, expt_model, expt_loss,
                                                         current_train,
                                                         dev_dataloader,
                                                         portion=fractions[i],
                                                         layers=list(layers).copy(),
                                                         eval_datasets = {'dev': dev_dataloader,
                                                                          'test': test_dataloader,
                                                                          'train': current_train})

        # Save evals and Load best model from this portion.

        if args['verbose']:
            tqdm.write(f"\nEvaluating best probe on dev and test...")

        for l in layers:
            online_coding_list[l].append(evals[l])
    
            expt_probe[l].load_state_dict(torch.load(evals[l]['probe_params_path']))
            expt_probe[l].eval()

            test_predictions = expt_regimen.predict(expt_probe[l], expt_model, test_dataloader)
            test_report = expt_reporter(test_predictions, test_dataloader, 'test')
            test_report_list[l].append(test_report)

            os.remove(evals[l]['probe_params_path'])


    # Final operations
    json.dump(online_coding_list, open(os.path.join(args['reporting']['root'], 'online_coding.json'), 'w'))
    json.dump(test_report_list, open(os.path.join(args['reporting']['root'], 'online_test_report.json'), 'w'))

    # Compute and save the metrics.
    return computeMDL(args, online_coding_list, test_report_list, layers)


def computeMDL(args, online_report, test_report, layers):

    num_classes = args['dataset']['label_space_size']
    train_size = online_report[layers[-1]][-1]['train_targets']
    uniform_codelength = train_size * np.log2(num_classes)

    results = {}

    for lvl in layers:
        
        lvl_online_codelength = online_report[lvl][0]['train_targets'] * np.log2(num_classes) + sum([elem['loss_online_portion'] for elem in online_report[lvl][:-1]])  

        lvl_compression = uniform_codelength / lvl_online_codelength
        lvl_test_acc = test_report[lvl][-1]['label_acc_test'] 
        
        lvl_f1_class_0 = test_report[lvl][-1]['f1_class_0_test']            
        lvl_f1_class_1 = test_report[lvl][-1]['f1_class_1_test']            

        if args['verbose']:
            tqdm.write(f"\n\nLayer: {lvl}")
            tqdm.write(f"\tUniform codelength: {round(train_size * np.log2(num_classes) / 1024, 4)} kbits")
            tqdm.write(f"\tOnline codelength: {round(lvl_online_codelength / 1024, 4)} kbits")
            tqdm.write(f"\tCompression: {round(lvl_compression, 4)}")
            tqdm.write(f"\tTest accuracy: {round(lvl_test_acc * 100, 2)}%")
            tqdm.write(f"\tF1-score 0-class: {round(lvl_f1_class_0, 2)}")
            tqdm.write(f"\tF1-score 1-class: {round(lvl_f1_class_1, 2)}")

        results[lvl] = {
            'uniform_codelength': uniform_codelength,
            'online_codelength': lvl_online_codelength,
            'compression': lvl_compression,
            'test_acc': lvl_test_acc,
            'f1_0_class_test': lvl_f1_class_0,
            'f1_1_class_test': lvl_f1_class_1
        }

    json.dump(results, open(os.path.join(args['reporting']['root'], 'mdl.json'), 'w'))

    return results




def run(exp_config):

    exp_config = os.path.join("../mdl_configs", exp_config)

    yaml_args = yaml.safe_load(open(exp_config))
    yaml_args["experiment_config"] = exp_config

    if yaml_args["seed"]:
        set_seed(yaml_args["seed"])

    print("Configuration readed!")
    # Single model probing.
    if "trajectories" not in yaml_args:
        yaml_args['verbose'] = True

        setup_new_experiment_dir(yaml_args)
        return execute_experiment(yaml_args)

    # Training checkpoints probing.
    else:
        yaml_args['verbose'] = False

        now = datetime.now()
        date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]))
        model_suffix = '-'.join((yaml_args['model']['type'], yaml_args['dataset']['name']))

        new_root = os.path.join(yaml_args['reporting']['root'], 'trajectories-' + date_suffix + '-' + model_suffix +'/')
        tqdm.write(f"Constructing trajectories directory at {new_root}")

        yaml_args['reporting']['root'] = new_root
        os.makedirs(new_root, exist_ok=True)

        # building steps
        steps = list(map(int, yaml_args['trajectories']['first_steps'].split(',')))
        total = yaml_args['trajectories']['total']

        while len(steps) < total:
            steps.append(steps[-1] + yaml_args['trajectories']['every'])

        for i, s in enumerate(steps):
            tqdm.write(f"[{datetime.now().time()}] Training probe on steps {s}: {i}/{len(steps)}")

            exp_yaml = copy.deepcopy(yaml_args)
            exp_yaml['model']['step'] = s

            setup_new_experiment_dir(exp_yaml)
            execute_experiment(exp_yaml)


if __name__ == '__main__':
    """Setup the arguments, load the experiment configuration, and execute the experiment."""

    argp = ArgumentParser()
    argp.add_argument('--experiment_config', type=str, help='Path to the experiment configuration file.')
    cli_args = argp.parse_args()

    run(cli_args.experiment_config)
