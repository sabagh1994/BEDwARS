import os, sys
import json
import torch
import numpy as np
import pandas as pd
from os.path import basename
curr_path = os.getcwd()
sys.path.append(curr_path)

from IO import MHLogger
from general import argsort_chains
from summary import simple_summary
from curric_utils import get_refsig_mix, ModelSaverLoader

##########################################################
#################### input ###############################
##########################################################

use_argparse = True
if use_argparse:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_config', action='store', type=str, required=True,
                        help='Path to config file containing the path to the input files'
                             ' such as reference signatures and mixtures, etc.')
    parser.add_argument('--curric_config', action='store', type=str, required=True,
                        help='Path to config file containing the curriculum used for Metropolis Hasting.')
    parser.add_argument('--stage_range', nargs="+", type=int,
                        help='Range of samples used for inference, '
                             'e.g., --stage_range 50 160 starts from 50 and ends at 160')
    parser.add_argument('--stage_name', action='store', type=str, required=True,
                        help="Name of the stage to be used for inference, e.g., L3")
    args = parser.parse_args()
    # get the curriculum config and the config containing the path to inout files
    general_config, curric_config = args.general_config, args.curric_config
    stage_name, stage_range = args.stage_name, args.stage_range

if not os.path.exists(general_config):
    raise FileNotFoundError(f"{general_config} does not exist!")

with open(general_config) as f:
    general_config_dict = json.load(f)

curric_config_name = basename(curric_config).split('.json')[0]
outdir = general_config_dict.get("outdir", "./results/")
root_storage_dir = f'{outdir}/{curric_config_name}'
model_storage_dir = f'{root_storage_dir}/models/'
aggr_storage_dir= f'{root_storage_dir}/aggr/'
os.makedirs(aggr_storage_dir, exist_ok=True)
print(f"config dir {general_config}\noutdir {outdir}")

###############################################################################
###################### Initializing Torch Device & Dtype ######################
###############################################################################
# collect device stats
print(torch.__version__)
print(f"cuda is available {torch.cuda.is_available()}")
print('Active CUDA Device: GPU', torch.cuda.current_device())
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
tch_dtype = torch.float32
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

np.set_printoptions(precision=2, suppress=True)

###############################################################################
################### Reading Mixture and Reference Signature ###################
###############################################################################

refsig_mix_dict = get_refsig_mix(general_config_dict, device, tch_dtype)
mixtures = refsig_mix_dict['mixtures']
sig_ref = refsig_mix_dict['sig_ref']
sig_org = refsig_mix_dict['sig_org']
props = refsig_mix_dict['props']
ct_names = refsig_mix_dict['ct_names']
ensid = refsig_mix_dict['ensid']
# related to mixtures, i.e., bulk profiles
mix_mean = refsig_mix_dict['mix_mean']  # to reverse mixture normalization
mixtures_org = refsig_mix_dict['mixtures_org']
mix_names = refsig_mix_dict['mix_names']

model_saver_loader = ModelSaverLoader(cache_size=1, sig_ref=sig_ref, device=device)
logfile = open(f'{aggr_storage_dir}/log_aggr', 'w')
logger = MHLogger(logfile)
logger(f'device: {device}\n')

###############################################################################
#################### Loading the models and aggregating samples ###############
###############################################################################

summ_dict, accept_ctr = {}, {}
org_var_dict = dict(sig_org=sig_org, sig_ref=sig_ref, props=props)

for model_ind in range(*stage_range):
    model0_path = f'{model_storage_dir}/{stage_name}-{model_ind}.pt'
    if not (os.path.exists(model0_path)):
        raise FileNotFoundError(f"{model0_path} does not exist!")
    model, MH_dict, _ = model_saver_loader.load(model0_path, map_location=device, torch_sum=torch_sum)

    for var in model.vars:
        var_value_untrans = var.backward_var(var.value)
        accept_ctr[var.name] = torch.tensor(stage_range[1] - stage_range[0])
        if not (var.name in summ_dict.keys()):
            summ_dict[var.name] = var_value_untrans
        else:
            summ_dict[var.name] = summ_dict[var.name] + var_value_untrans

mean_dict = simple_summary(model, summ_dict, accept_ctr, org_var_dict)
# getting S_hat as mean_dict only contains estimation of B and sigma_b
sigma_b_hat, B_hat, = mean_dict['sigma_b'], mean_dict['B']
S_hat, W_hat = mean_dict['S'], mean_dict['W']

# sorting the chains based on the criterion of interest
sort_criterion = "MSE_marker"  # posterior, LL, MSE_marker
sort_inds, criterion = argsort_chains(sort_criterion, model, mixtures, point=mean_dict,
                                      is_transformed=False, extra_dict={'FCs': [4.]})

best_chain_id = sort_inds[0].cpu()[0].item()
S_hat_sorted = S_hat[sort_inds[0].cpu(), :]
W_hat_sorted = W_hat[sort_inds[0].cpu(), :]

W_best = W_hat_sorted[0].cpu().numpy()
S_best = S_hat_sorted[0].cpu().numpy()
S_best_norm = (S_best - np.mean(S_best, axis=0))/np.var(S_best, axis=0)**0.5

# get the estimated bulk expression
mix_mean = mix_mean.cpu().numpy()
# X_norm should be compared with the preprocessed mixtures used in deconvolution
X_norm = S_best@W_best
# X_best and X_org should be compared
X_best = X_norm*mix_mean.reshape((1, mix_mean.size))  # (G, N) # reversing the normalization in preprocessing
X_org = mixtures_org.cpu().numpy()

# get the estimated bulk expression per cell type
G, C = S_best.shape
_, N = X_best.shape
S_best_re = S_best.T.reshape((C, G, 1))  # C*G => to reshape into C*G*1
W_best_re = W_best.reshape((C, 1, N))
mix_mean_re = mix_mean.reshape((1, 1, N))
X_best_CT = (S_best_re*W_best_re)*mix_mean_re  # shape (C, G, N)

# storing the best estimated signature and proportions
np.savez(f"{aggr_storage_dir}/BEDwARS_estim.npz", W=W_best, S=S_best,
         S_norm=S_best_norm, ct_names=ct_names, ensid=ensid,
         mix_mean=mix_mean, mix_names=mix_names, X_best=X_best, X_best_CT=X_best_CT)
# save dataframes
df_S = pd.DataFrame(S_best, columns=ct_names, index=ensid)
df_S_norm = pd.DataFrame(S_best_norm, columns=ct_names, index=ensid)
df_W = pd.DataFrame(W_best, columns=mix_names, index=ct_names)
df_S.to_csv(f"{aggr_storage_dir}/signatures", sep="\t")
df_S_norm.to_csv(f"{aggr_storage_dir}/signatures_norm", sep="\t")
df_W.to_csv(f"{aggr_storage_dir}/proportions", sep="\t")

os.makedirs(f"{aggr_storage_dir}/bulk_CT", exist_ok=True)
for ct_ind, ct_name in enumerate(ct_names):
    df_ct = pd.DataFrame(X_best_CT[ct_ind], index=ensid)
    df_ct.columns = mix_names
    df_ct.to_csv(f"{aggr_storage_dir}/bulk_CT/{ct_name}", sep="\t")