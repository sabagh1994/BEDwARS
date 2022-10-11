import os
import torch
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr


def celltype_stats(true_param, estim_param, is_sig=None):
    """
    Computes various statistics on the estimated and true parameters e.g., signatures or proportions per cell type

    :param true_param: (torch.Tensor) true parameter
    :param estim_param: (torch.Tensor) estimated parameter
    :param is_sig: (boolean) whether the input variable is signature
    :return: (tuple) computed statistics
    """

    assert true_param.shape[-1] == estim_param.shape[-1], "inconsistent cell type dimension"
    corrs, spears, rmses, nrmses, maes, nmaes = [], [], [], [], [], []
    C = true_param.shape[-1]  # the last dimension must be cell type
    for ct_ind in range(C):
        true = true_param[:, ct_ind].reshape(-1)
        est = estim_param[:, ct_ind].reshape(-1)
        if is_sig:
            # for valid rmse normalization should be done
            true = ((true - torch.mean(true))/(torch.var(true)**0.5 + 0.00001))
            est = ((est - torch.mean(est))/(torch.var(est)**0.5 + 0.00001))
        true, est = true.cpu(), est.cpu()  # for spearman and pearson should be on cpu
        
        if torch.any(torch.isnan(true)) or torch.any(torch.isnan(est)):
            corr, spear = -1, -1
        else:
            corr, _ = pearsonr(true, est)
            spear, _ = spearmanr(true, est)
        rmse = torch.mean((true - est)**2)**0.5
        nrmse = rmse/torch.mean(true)  # not valid for normalized signatures
        mae = torch.mean(abs(est - true))
        nmae = mae/torch.mean(true)  # not valid for normalized signatures

        corrs.append(corr)
        spears.append(spear)
        rmses.append(rmse)
        nrmses.append(nrmse)
        maes.append(mae)
        nmaes.append(nmae)
    return corrs, spears, rmses, nrmses, maes, nmaes


def simple_summary(model, summ_dict, accept_ctr, org_var_dict):
    """
    This summary has no computation of statistics such as MSE, MAE, ... for the inferred variables
    The output is just the inferred values of variables

    :param model: (Model.Model) Model object
    :param summ_dict: (dict) Running sum of the variables
    :param accept_ctr: (dict) The number of times each variable was accepted
    :param org_var_dict: (dict) Contains reference signature, original signature and proportions if available
                         original signature and proportions are used for development purposes
    :return mean_dict: (dict) Average of accepted samples per variable, i.e. inferred value of each variable
    """
    mean_dict = {}
    for var in model.vars:
        acc_var = accept_ctr[var.name]
        acc_var = acc_var + (acc_var == 0).to(acc_var)
        mean_dict[var.name] = summ_dict[var.name]/acc_var

    sig_ref = org_var_dict.get("sig_ref", None)
    assert not(sig_ref is None), "reference signature not available for summarizing"
    G, C = sig_ref.shape

    if not model.fixed_sigs:
        B_hat = mean_dict['B']  # (chain_size, G, C)
        sigma_b_hat = mean_dict['sigma_b']
    assert B_hat.shape == (*model.chain_size, G, C), f"B_hat shape {B_hat.shape}"

    if not model.fixed_sigs:
        u = sig_ref
        u = u.reshape((*model.dim_ext, G, C))
        cv = u * (sigma_b_hat**2)
        cv = torch.sqrt(torch.abs(cv))
        S_hat = u + (B_hat*cv)

        if model.transformation == "log":
            S_hat = torch.exp(S_hat)
    else:
        S_hat = sig_ref
        S_hat = S_hat.reshape((*model.dim_ext, G, C))  # (1, G, C)

    mean_dict["S"] = S_hat
    return mean_dict


def compute_summary_multichain(model, X_obs, summ_dict, accept_ctr, org_var_dict, torch_sum=None):
    """
    Summarize the samples by taking the average of accepted samples. Compute statistics between the
    true and estimated variables if the ground truth (original signatures or proportions) is available

    :param model: (Model.Model) Model object
    :param X_obs: (torch.Tensor) Bulk expression profiles
    :param summ_dict: (dict) Running sum for each variable. Sum of the samples taken for the variable
                      that were accepted. Keys are variable names.
    :param accept_ctr: (dict) Count of times the proposed update was accepted for the variable.
                        Keys are variable names.
    :param org_var_dict: (dict) Contains reference signature, original signature and proportions if available.
                         Original signature and proportions are used for development purposes
    :param torch_sum: Function to perform summation
    :return: (tuple)
    """
    '''
        Note: posterior and LL calculation have not been tested in this function
    '''
    mean_dict = {}
    for var in model.vars:
        acc_var = accept_ctr[var.name]
        acc_var = acc_var + (acc_var == 0).to(acc_var)
        mean_dict[var.name] = summ_dict[var.name]/acc_var

    MSE_dict = {}
    original_posterior = final_posterior = 0.
    # MSE computations:
    props = org_var_dict.get("props", None)
    sig_ref, sig_org = org_var_dict.get("sig_ref", None), org_var_dict.get("sig_org", None)
    assert not(sig_ref is None), "reference signature not available for summarizing"
    G, C = sig_ref.shape

    if not model.fixed_sigs:
        B_hat = mean_dict['B']  # (chain_size, G, C)
        sigma_b_hat = mean_dict['sigma_b']
    assert B_hat.shape == (*model.chain_size, G, C), f"B_hat shape {B_hat.shape}"

    if not model.fixed_sigs:
        u = sig_ref
        u = u.reshape((*model.dim_ext, G, C))
        cv = u * (sigma_b_hat**2)
        cv = torch.sqrt(torch.abs(cv))
        S_hat = u + (B_hat*cv)

        if model.transformation == "log":
            S_hat = torch.exp(S_hat)
    else:
        S_hat = sig_ref
        S_hat = S_hat.reshape((*model.dim_ext, G, C))  # (1, G, C)

    # signature MSE total and per cell type, corrs per (chain, cell type)
    if not model.fixed_sigs:
        if not(sig_org is None):
            S_MSE = torch.mean((sig_org - S_hat) ** 2, dim=(-2, -1), keepdim=True)  # (chain_size, 1, 1)
            MSE_dict['S'] = S_MSE
            MSE_dict[f'S_CT'] = torch.mean((sig_org - S_hat) ** 2,
                                           dim=-2, keepdim=True)  # (chain_size, 1, C) => over the genes

            chain_stats_dict = defaultdict(list)
            for chain_ind in range(S_hat.shape[0]):
                S_hat_chain = S_hat[chain_ind, :, :]
                stat_tuple = celltype_stats(true_param=sig_org, estim_param=S_hat_chain, is_sig=True)
                corrs, spears, rmses, nrmses, maes, nmaes = stat_tuple

                chain_stats_dict['chain_stats_corr'].append(corrs)
                chain_stats_dict['chain_stats_spear'].append(spears)
                chain_stats_dict['chain_stats_rmse'].append(rmses)
                chain_stats_dict['chain_stats_nrmse'].append(nrmses)
                chain_stats_dict['chain_stats_mae'].append(maes)
                chain_stats_dict['chain_stats_nmae'].append(nmaes)
            assert np.array(chain_stats_dict['chain_stats_corr']).shape == (*model.chain_size, C), "mismatch in shape in summary"
            MSE_dict[f'S_corr_CT'] = np.array(chain_stats_dict['chain_stats_corr']) # (chain_size, C)
            MSE_dict[f'S_spear_CT'] = np.array(chain_stats_dict['chain_stats_spear'])
            MSE_dict[f'S_rmse_CT'] = np.array(chain_stats_dict['chain_stats_rmse'])  # (chain_size, C)
            MSE_dict[f'S_nrmse_CT'] = np.array(chain_stats_dict['chain_stats_nrmse'])  # (chain_size, C)
            MSE_dict[f'S_mae_CT'] = np.array(chain_stats_dict['chain_stats_mae'])
            MSE_dict[f'S_nmae_CT'] = np.array(chain_stats_dict['chain_stats_nmae'])

    W_hat = mean_dict['W']
    sigma_x_hat = mean_dict['sigma_x']
    alpha_hat = mean_dict['alpha']
    if not(props is None):
        W_MSE = torch.mean((W_hat - props)**2, dim=(-2, -1), keepdim=True)  # (chain_size, 1, 1)
        MSE_dict['W'] = W_MSE
        MSE_dict[f'W_CT'] = torch.mean((W_hat - props)**2,
                                       dim=-1, keepdim=True)  # (chain_size, C, 1) => over the samples
        chain_stats_dict = defaultdict(list)
        for chain_ind in range(W_hat.shape[0]):
            W_hat_chain = W_hat[chain_ind, :, :]
            stat_tuple = celltype_stats(true_param=props.T, estim_param=W_hat_chain.T, is_sig=False)
            corrs, spears, rmses, nrmses, maes, nmaes = stat_tuple

            chain_stats_dict['chain_stats_corr'].append(corrs)
            chain_stats_dict['chain_stats_spear'].append(spears)
            chain_stats_dict['chain_stats_rmse'].append(rmses)
            chain_stats_dict['chain_stats_nrmse'].append(nrmses)
            chain_stats_dict['chain_stats_mae'].append(maes)
            chain_stats_dict['chain_stats_nmae'].append(nmaes)

        MSE_dict[f'W_corr_CT'] = np.array(chain_stats_dict['chain_stats_corr'])  # (chain_size, C)
        MSE_dict[f'W_spear_CT'] = np.array(chain_stats_dict['chain_stats_spear'])
        MSE_dict[f'W_rmse_CT'] = np.array(chain_stats_dict['chain_stats_rmse'])  # (chain_size, C)
        MSE_dict[f'W_nrmse_CT'] = np.array(chain_stats_dict['chain_stats_nrmse'])  # (chain_size, C)
        MSE_dict[f'W_mae_CT'] = np.array(chain_stats_dict['chain_stats_mae'])
        MSE_dict[f'W_nmae_CT'] = np.array(chain_stats_dict['chain_stats_nmae'])

    X_hat = S_hat@W_hat
    X_MSE = torch.mean((X_obs - X_hat)**2, dim=(-2, -1), keepdim=True)  # (chain_size, 1, 1)
    MSE_dict['X'] = X_MSE

    original_posterior = None
    point = {'sigma_x': torch.log(sigma_x_hat),
             'alpha': model.alpha.distribution.transform.forward(alpha_hat),
             'W': model.W.distribution.transform.forward(W_hat)}

    if not model.fixed_sigs:
        point['sigma_b'] = torch.log(sigma_b_hat)
        point['B'] = B_hat

    final_posterior = model.logp(observed=X_obs, point=point, torch_sum=torch_sum)  # (chain_size, 1, 1)
    return mean_dict, MSE_dict, (original_posterior, final_posterior)


def save_summaries_multichain(outdir, record_ind, summ_dict):
    os.makedirs(outdir, exist_ok=True)
    torch.save(summ_dict, f'{outdir}/stats{record_ind}')  # make sure it can load even if it is not on cuda


def merge_summ_multichain(summ_dict1, summ_dict2):
    """
        Merges two summary dicts over variable
    """
    summ_dict_cat = dict()
    for var_name in summ_dict1.keys():
        val1, val1_accept = summ_dict1[var_name]
        val2, val2_accept = summ_dict2[var_name]
        val = torch.cat([val1, val2], dim=0)
        val_accept = torch.cat([val1_accept, val2_accept], dim=0)
        summ_dict_cat[var_name] = (val, val_accept)

    return summ_dict_cat


def subsetter_summ_multichain(summ_dict, chain_inds):
    """
        Subsets the chain_inds out of summ_dict
    """
    summ_dict_sub = dict()
    for var_name in summ_dict.keys():
        val, val_accept = summ_dict[var_name]
        summ_dict_sub[var_name] = (val[chain_inds, :, :], val_accept[chain_inds, :, :])
    return summ_dict_sub
