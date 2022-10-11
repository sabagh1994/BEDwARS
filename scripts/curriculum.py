import os, sys
import json
import torch
import numpy as np
from os.path import basename, exists
curr_path = os.getcwd()
sys.path.append(curr_path)

from curric_utils import (model_primary_init, MH_primary_init,
                          get_refsig_mix, ModelSaverLoader, get_rng_seeds_arr)
from general import argsort_chains
from MH import subsetter_MH, random_walk, merge_MH
from summary import subsetter_summ_multichain, merge_summ_multichain
from curric_utils import preprocess_curric_config
from IO import MHLogger
from Model import Model

np.set_printoptions(precision=2, suppress=True)

use_argparse = True
if use_argparse:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_config', action='store', type=str, required=True,
                        help='Path to config file containing the path to the input files'
                             ' such as reference signatures and mixtures, etc.')
    parser.add_argument('--curric_config', action='store', type=str, required=True,
                        help='Path to config file containing the curriculum used for Metropolis Hasting.')
    args = parser.parse_args()
    # get the curriculum config and the config containing the path to inout files
    general_config, curric_config = args.general_config, args.curric_config

###############################################################################
###################### Initializing Torch Device & Dtype ######################
###############################################################################

# collect device stats
print(torch.__version__)
print(torch.cuda.is_available())
print('Active CUDA Device: GPU', torch.cuda.current_device())
print(torch.cuda.device_count())  # gets the number of GPUs
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
tch_dtype = torch.float32
print(torch.get_num_threads())
use_cupy_sum = True

if use_cupy_sum:
    import cupy as cp
    from torch.utils.dlpack import to_dlpack
    from torch.utils.dlpack import from_dlpack

    def torch_sum(input, dim, keepdim):
        if input.dtype == torch.bool:
            return torch.sum(input, dim=dim, keepdim=keepdim)
        else:
            cp_input = cp.fromDlpack(to_dlpack(input))
            cp_sum = cp.sum(cp_input, axis=dim, keepdims=keepdim)
            return from_dlpack(cp_sum.toDlpack())
else:
    torch_sum = torch.sum
###############################################################################
################### Reading Mixture and Reference Signature ###################
###############################################################################

with open(general_config) as f:
    general_config_dict = json.load(f)
with open(curric_config) as f:
    curric_config_dict = json.load(f)

refsig_mix_dict = get_refsig_mix(general_config_dict, device, tch_dtype)
mixtures = refsig_mix_dict['mixtures']
sig_ref = refsig_mix_dict['sig_ref']
sig_org = refsig_mix_dict['sig_org']
props = refsig_mix_dict['props']

###############################################################################
######################### Creating Empty Directories ##########################
###############################################################################

curric_config_name = basename(curric_config).split('.json')[0]
outdir = general_config_dict.get("outdir", "./results/")
root_storage_dir = f'{outdir}/{curric_config_name}'
model_storage_dir = f'{root_storage_dir}/models/'
os.makedirs(outdir, exist_ok=True)
os.makedirs(model_storage_dir, exist_ok=True)

model_saver_loader = ModelSaverLoader(cache_size=1, sig_ref=sig_ref, device=device)
logfile = open(f'{root_storage_dir}/status', 'w')
logger = MHLogger(logfile)
logger(f'device: {device}\n')

###############################################################################
############################ Curriculum Stages ################################
###############################################################################

# Repeating MCMC steps and whatnot
curric_config_dict_post = preprocess_curric_config(curric_config_dict)

for stage_dict in curric_config_dict_post:
    curc_stage_name = stage_dict['stage_name']
    curc_operation = stage_dict['stage_oper']
    curc_stage_args = stage_dict['stage_args']

    assert curc_operation in ('init', 'reseed', 'concat', 'sort_subset', 'mcmc')
    logger(f"stage_name {curc_stage_name}, curc_operation {curc_operation}, "
           f"curc_stage_args {curc_stage_args}\n", stdout=True)
    logger.flush()

    #######################################
    ########### Stage Variables ###########
    #######################################
    stage_model_path = f'{model_storage_dir}/{curc_stage_name}.pt'

    # # ATTENTION: this has to be handled better! had to disable for debugging
    if exists(stage_model_path):  # weak fault-tolerancy
        continue

    #######################################
    ########## Stage Operations ###########
    #######################################
    if curc_operation == 'init':
        assert set(curc_stage_args.keys()) == {'rng_seeds', 'model_config', 'chain_config'}

        # Getting the stage arguments
        seed_picking_method, seed_picking_args = curc_stage_args['rng_seeds']
        model_config_dict, chain_config_dict = curc_stage_args['model_config'], curc_stage_args['chain_config']
        w0_dir = model_config_dict.get('w0_dir', None)
        model_config_dict["w0"] = np.load(model_config_dict['w0_dir'])["w0"] if not(w0_dir is None) else None

        # Creating the rng_seeds array
        rng_seeds = get_rng_seeds_arr(seed_picking_method, seed_picking_args, model_config_dict['chains'])
        transformation = general_config_dict.get("transformation", "log")
        model = model_primary_init(model_config_dict, chain_config_dict, sig_ref,
                                   mixtures, transformation, tch_dtype, device, rng_seeds, torch_sum)
        MH_dict = MH_primary_init(model, model_config_dict)
        summ_dict = None
    elif curc_operation == 'reseed':
        assert set(curc_stage_args.keys()) == {'past_stage_name', 'rng_seeds'}

        # Getting the stage arguments
        old_stage_name = curc_stage_args['past_stage_name']
        seed_picking_method, seed_picking_args = curc_stage_args['rng_seeds']

        # Loading the old model
        old_model_path = f'{model_storage_dir}/{old_stage_name}.pt'
        model, MH_dict, summ_dict = model_saver_loader.load(old_model_path, map_location= device, torch_sum= torch_sum)

        # Creating the rng_seeds array, Reseeding the model
        new_rng_seeds = get_rng_seeds_arr(seed_picking_method, seed_picking_args, np.prod(model.chain_size))
        model.sampler.seed(new_rng_seeds)  # assuming that the sampler for MH_dict is also updated
    elif curc_operation == 'concat':
        # get two stage names or a list
        # create the model
        # feed to merge

        # merging of two models should be done after they have been running for the same number of steps
        assert set(curc_stage_args.keys()) == {'past_stage_names'}
        stage_names = curc_stage_args['past_stage_names']  # list of stage names

        model0_path = f'{model_storage_dir}/{stage_names[0]}.pt'
        model, MH_dict, summ_dict = model_saver_loader.load(model0_path, map_location=device, torch_sum=torch_sum)

        for sec_stage_name in stage_names[1:]:
            model2_path = f'{model_storage_dir}/{sec_stage_name}.pt'
            model2, MH_dict2, summ_dict2 = model_saver_loader.load(model2_path, map_location=device, torch_sum=torch_sum)
            model = Model.merge_models(model, model2, torch_sum)
            MH_dict = merge_MH(MH_dict, MH_dict2, model)
            summ_dict = merge_summ_multichain(summ_dict, summ_dict2)
    elif curc_operation == 'sort_subset':
        assert set(curc_stage_args.keys()) == {'past_stage_name',
                                               'sort_criterion', 'extra_dict', 'use_summary',
                                               'is_transformed', 'topk', 'chain_inds_range'}
        old_stage_name = curc_stage_args['past_stage_name']
        sort_criterion = curc_stage_args['sort_criterion']
        extra_dict = curc_stage_args['extra_dict'] # example of extra_dict= {'FCs': [5.]}, extra_dict= {'FCs': [4., 5.]}
        topk = curc_stage_args['topk']
        use_summary = curc_stage_args['use_summary']
        is_transformed = curc_stage_args['is_transformed'] # whether the provided point is transformed or not
        assert isinstance(use_summary, bool)
        assert (use_summary and not(is_transformed)) or (not(use_summary) and is_transformed)  # use_summary being True should be used with is_transformed being False
        chain_inds_range = curc_stage_args['chain_inds_range']

        # Loading the old model
        old_model_path = f'{model_storage_dir}/{old_stage_name}.pt'
        model, MH_dict, summ_dict = model_saver_loader.load(old_model_path, map_location=device, torch_sum= torch_sum)

        if (chain_inds_range is None):
            if use_summary:
                # the summ_dict saved is a dictionary of var.name as key and (var_sum, var_accept) as values
                point = {}
                is_transformed= False
                for var_name, (var_sum, var_accept) in summ_dict.items():
                    point[var_name] = var_sum/var_accept
            else:
                point = None

            # Sorting the chains and producing their indices
            sorted_all_chain_inds, _ = argsort_chains(sort_criterion, model, X=mixtures,
                                                     point=point, is_transformed=is_transformed,
                                                     extra_dict=extra_dict, torch_sum=torch_sum)
            # Subsetting the (top) chain indices   # Attention: subsetting based on multiple FC then taking the union
            chain_inds = [inds[:topk] for inds in sorted_all_chain_inds]
            chain_inds = torch.unique(torch.cat(chain_inds)).reshape(-1)
        else:
            chain_inds = list(range(*chain_inds_range))
        print(f'chain_inds is {chain_inds}')

        # Subsetting the chains
        model = Model.subsetter_model(model, chain_inds, torch_sum)
        MH_dict = subsetter_MH(MH_dict, model, chain_inds)
        summ_dict = subsetter_summ_multichain(summ_dict, chain_inds)
    elif curc_operation == 'mcmc':
        assert set(curc_stage_args.keys()) == {'past_stage_name',
            'steps', 'after_tune', 'get_stats', 'log_acc_rate',
            'log_interval', 'log_var_vals'}
        old_stage_name = curc_stage_args['past_stage_name']
        steps = curc_stage_args['steps']
        after_tune = curc_stage_args['after_tune']
        get_stats = curc_stage_args['get_stats']
        log_acc_rate = curc_stage_args['log_acc_rate']
        log_interval = curc_stage_args['log_interval']
        log_var_vals = curc_stage_args['log_var_vals']

        assert isinstance(get_stats, bool)
        assert isinstance(log_acc_rate, bool)
        assert isinstance(log_var_vals, bool)

        # load the model and MH_dict to perform on
        # Loading the old model
        old_model_path = f'{model_storage_dir}/{old_stage_name}.pt'
        model, MH_dict, _ = model_saver_loader.load(old_model_path, map_location=device, torch_sum= torch_sum)

        # Making sure no tunning goes to waste!
        if not(after_tune):
            for key, mh_obj in MH_dict.items():
                assert steps % (mh_obj.tune_interval) == 0

        org_var_dict = dict(sig_org=sig_org, sig_ref=sig_ref, props=props)
        summ_dict = random_walk(model, MH_dict, mixtures, steps, after_tune, org_var_dict, get_stats,
                                log_acc_rate, log_var_vals, torch_sum, logger, log_interval)
    else:
        raise ValueError(f'Operation {curc_operation} not implemented.')

    #######################################
    ########## Storing the Model ##########
    #######################################
    model_saver_loader.save(stage_model_path, model, MH_dict, summ_dict)

