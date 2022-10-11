import copy
from collections import OrderedDict

import torch
import numpy as np

from preprocessing import preprocessor
from Model import Model
from MH import set_MH_state, get_MH_state
from rng import BatchCudaRNG, SerialRNG


def get_refsig_mix(general_config_dict, device, tch_dtype):
    """
    Preprocess the signatures and bulk expression profiles
    """
    # get general configs for preprocessing
    naive_marker_FC = general_config_dict.get("naive_marker_FC", 2.0)
    marker_type = general_config_dict.get("marker_type", "all")
    marker_dir = general_config_dict.get("marker_dir", None)
    normalization = general_config_dict.get("normalization", "mean")
    transformation = general_config_dict.get("transformation", "log")
    ref_sig_dir = general_config_dict.get("ref_sig_dir", None)
    org_sig_dir = general_config_dict.get("org_sig_dir", None)
    mix_dir = general_config_dict.get("mix_dir", None)
    org_prop_dir = general_config_dict.get("org_prop_dir", None)
    assert (mix_dir and ref_sig_dir), "path for mixture/reference signature is missing!"
    marker_dict = dict(marker_type=marker_type, marker_dir=marker_dir,
                       naive_marker_FC=naive_marker_FC)

    pkg = preprocessor(ref_sig_dir=ref_sig_dir, mix_dir=mix_dir,
                       org_sig_dir=org_sig_dir, org_prop_dir=org_prop_dir,
                       marker_dict=marker_dict, transformation=transformation,
                       normalization=normalization)
    mixtures, sig_ref, sig_org, props, ct_names, ensid, mix_mean, mixtures_org, mix_names = pkg
    mixtures = torch.from_numpy(mixtures).to(device, tch_dtype)
    sig_ref = torch.from_numpy(sig_ref).to(device, tch_dtype)
    sig_org = torch.from_numpy(sig_org).to(device, tch_dtype) if not(sig_org is None) else None
    props = torch.from_numpy(props).to(device, tch_dtype) if not(props is None) else None

    mixtures_org = torch.from_numpy(mixtures_org).to(device, tch_dtype)
    mix_mean = torch.from_numpy(mix_mean).to(device, tch_dtype) if not(mix_mean is None) else None

    out_dict = dict(mixtures=mixtures, sig_ref=sig_ref, sig_org=sig_org,
                    props=props, ct_names=ct_names, ensid=ensid,
                    mix_mean=mix_mean, mixtures_org=mixtures_org, mix_names=mix_names)
    return out_dict


def model_primary_init(model_config_dict, chain_config_dict, sig_ref, mixtures,
                       transformation, tch_dtype, device, rng_seeds, torch_sum):
    """
    Initialize the model and the sampler used for it for the first time using the model config
    Initialize the variables of the model using the chain config

    :param model_config_dict: (dict) contains the distribution parameters for all variables in the model
    :param chain_config_dict: (dict) contains the mode of initialization ("prior", "default", "manual")
                              for the variables. "manual" initialization mode is not implemented yet.
                              It is best to stick with the initializations used for paper experiments.
    :param sig_ref: (torch.Tensor) reference signature
    :param mixtures: (torch.Tensor) bulk expression profiles
    :param transformation: (str) "log" transformation is used by default
    :param tch_dtype: (torch.float32) dtype of the torch tensors
    :param device: (torch.device) to run the experiments on
    :param rng_seeds: (np.ndarray) array of random number generator (rng) seeds used for the sampler
    :param torch_sum: method used for summation
    :return model: (Model.Model) instance of the Model with its variables and sampler initialized
    """

    # init model
    chains = model_config_dict.get("chains", 20)
    use_batch_rng = model_config_dict.get("use_batch_rng", True)
    chain_size = (chains,)
    # set the rng, sampler
    if use_batch_rng:
        assert isinstance(rng_seeds, np.ndarray), f"{type(rng_seeds)} is not an array type!"
        assert rng_seeds.size == chains, f'{rng_seeds.size} != {chains}'
        sampler = BatchCudaRNG(device, chain_size, tch_dtype)
        sampler.seed(rng_seeds.reshape(chain_size))
    else:
        assert isinstance(rng_seeds, int), "seed instance is not correct"
        sampler = SerialRNG()
        torch.manual_seed(rng_seeds)

    eps = model_config_dict.get("eps", 1e-9)  # for dirichlet
    fixed_sigs = model_config_dict.get("fixed_sigs", False)
    props_alpha_distr = model_config_dict.get("props_alpha_distr", "uniform")
    alpha_lower = model_config_dict.get("alpha_lower", 0.0)
    alpha_upper = model_config_dict.get("alpha_upper", 30.0)
    sigma_b_distr = model_config_dict.get("sigma_b_distr", "halfcauchy")
    sigma_x_distr = model_config_dict.get("sigma_x_distr", "halfcauchy")
    beta_b = model_config_dict.get("beta_b", 1.0)
    beta_x = model_config_dict.get("beta_x", 5.0)

    model_param_setting = {'props_alpha': (props_alpha_distr,
                                           {'lower': torch.tensor(alpha_lower).to(device, tch_dtype),
                                            'upper': torch.tensor(alpha_upper).to(device, tch_dtype)}),
                           'sig_sigma': (sigma_b_distr, {'beta': torch.tensor(beta_b).to(device, tch_dtype)}),
                           'x_sigma': (sigma_x_distr, {'beta': torch.tensor(beta_x).to(device, tch_dtype)})}

    # extra model setting
    G, C = sig_ref.shape
    N = mixtures.shape[1]
    dim_ext = [1]*len(chain_size)
    # modules for modular update of signature (removed)
    w0 = model_config_dict["w0"]
    if w0 is None:
        w0 = 1./C * np.ones(C)
    w0 = w0.reshape(-1, 1)/np.sum(w0)  # sum to 1
    w0 = torch.from_numpy(w0).to(device, tch_dtype)
    model_extra_setting = dict(C=C, G=G, N=N, eps=eps, w0=w0, sig_ref=sig_ref,
                               fixed_sigs=fixed_sigs, transformation=transformation,
                               sampler=sampler, chain_size=chain_size)

    # create the model
    # initialize the variables
    # get chain configs for parameter initializations
    model = Model(model_param_setting, model_extra_setting)
    # default: first initialize all variables with priors
    model.initialization(seed=None, init_values={},
                         do_prior_init_dict={}, torch_sum=torch_sum)

    for var in model.vars:
        if (var.name == "W") or (var.name == "B"):
            init_mode = chain_config_dict.get(var.name, "default")
        elif (var.name == "sigma_b") or (var.name == "sigma_x") or (var.name == "alpha"):
            init_mode = chain_config_dict.get(var.name, "prior")
        else:
            raise ValueError("invalid varname")
        # initializing the variables
        if init_mode == "default":
            if var.name == "W":
                w0 = w0.reshape((*dim_ext, 1, C))
                # W size (chain_size, C, N), w0 size (1, 1, C)
                W_init = w0.transpose(dim0=-1, dim1=-2).repeat((*chain_size, 1, N))
                model.W.value = model.W.distribution.transform.forward(W_init)
            elif var.name == 'B':
                if not fixed_sigs:
                    model.__dict__[var.name].value = torch.zeros((*chain_size, G, C), device=device, dtype=tch_dtype)
            elif var.name == 'sigma_b':
                delta = np.prod(chain_size) // 4
                initial_sigma_b = model.__dict__[var.name].distribution.transform.backward(model.__dict__[var.name].value)
                initial_sigma_b[:delta, :, :] = 0.01
                initial_sigma_b[delta:2*delta, :, :] = 0.1
                initial_sigma_b[2*delta:3*delta, :, :] = 1.
                model.__dict__[var.name].value = model.__dict__[var.name].distribution.transform.forward(initial_sigma_b)
            elif var.name == 'alpha':
                # default for alpha uses prior
                pass
            elif var.name == 'sigma_x':
                # default for sigma_x uses prior
                pass
            else:
                raise ValueError("invalid variable name")
        elif init_mode == "prior":
            pass
        elif init_mode == "manual":
            raise ValueError("not implemented error")
            # # not fixed yet
            # init_vals= np.load(f"{chain_config_root}/chain{chain_id}/chain_init.npz")
            # assert var.name in init_vals.keys(), f"{var.name} manual initialization not provided"
            # init_val= torch.tensor(init_vals[var.name]).to(device, tch_dtype)
            # model.__dict__[var.name].value= model.__dict__[var.name].distribution.transform.forward(init_val) if var.is_transform else init_val
    _ = model.logp_modif(observed=mixtures, point=None, in_update=False,
                         overwrite=True, torch_sum=torch_sum)
    return model


def MH_primary_init(model, model_config_dict):
    """
    Model should be initialized first then fed into this function
    Primary initialization of Metropolis Hasting (MH) for each variable of the model
    :param model: (Model.Model) instance of the model
    :param model_config_dict: (dict) Model configuration
    :return MH_dict: (dict) contains MH (MH.Metropolis) for each variable of the model
                     with variable names as its keys.
    """

    MH_state = {}
    scaling_dict = {}
    device, tch_dtype = model.device, model.tch_dtype
    divide_B, chain_size = model_config_dict['divide_B'], model.chain_size
    for var in model.vars:
        if var.name == 'B' and divide_B:
            scaling = torch.ones((*chain_size, 1, model.C), device=device,
                                 dtype=tch_dtype)  # torch.tensor(1., device= device)
        else:
            scaling = torch.ones((*chain_size, 1, 1), device=device,
                                 dtype=tch_dtype)  # torch.tensor(1., device= device)
        scaling_dict[var.name] = scaling

    MH_state['scaling_dict'] = scaling_dict
    MH_state['divide_B'] = model_config_dict.get("divide_B", True)
    MH_state['tune_interval'] = model_config_dict.get("tune_interval", 50)
    MH_state['tune'] = True

    MH_dict = set_MH_state(model, MH_state)
    return MH_dict


class ModelSaverLoader:
    def __init__(self, cache_size, sig_ref, device):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.sig_ref = sig_ref
        self.device = device

    def save(self, save_path, model, MH_dict, summ_dict):
        """
        Saves the model and MH states as well as the summary of the variables
        Also, a copy is saved to the cache for faster loading later.

        :param save_path: (str) path to save the model
        :param model: (Model.Model) model to be saved
        :param MH_dict: (dict) MH object (MH.Metropolis) for all variables
        :param summ_dict: (dict) running sum of the variables and their acceptance rate. Check
                          the output of MH.random_walk() for more details.
        :return:
        """

        MH_state = get_MH_state(MH_dict)
        model_state = model.get_model_state()
        save_dict = dict(MH_state=MH_state, model_state=model_state,
                         summ_dict=summ_dict)
        torch.save(save_dict, save_path)

        self.cache[save_path] = save_dict  # (model, MH_dict)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def load(self, load_path, map_location, torch_sum):
        """
        Load the model and MH state and summary of variables to create a new model
        and MH objects for variables.

        :param load_path: (str) The path to load from
        :param map_location: Argument to torch.load to define the device to load on
        :param torch_sum: Function to perform summation
        :return: (tuple) Contains a model with its state set to the loaded model state,
                MH object for each variable, and the summary of the variables.
        """
        if load_path in self.cache:
            load_dict = self.cache[load_path]
            load_dict = copy.deepcopy(load_dict)
        else:
            load_dict = torch.load(load_path, map_location=map_location)
        model_state = load_dict['model_state']
        MH_state = load_dict['MH_state']
        summ_dict = load_dict['summ_dict']
        model = Model.set_model_state(model_state, self.sig_ref, self.device, torch_sum)
        MH_dict = set_MH_state(model, MH_state)

        return model, MH_dict, summ_dict

    def save_debug(self, save_path, model, MH_dict, summ_dict):
        MH_state = get_MH_state(MH_dict)
        model_state = model.get_model_state()
        save_dict = dict(MH_state=MH_state, model_state=model_state,
                         summ_dict=summ_dict)
        torch.save(save_dict, save_path)

        self.cache[save_path] = (model_state, MH_state, summ_dict, MH_dict)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def load_debug(self, load_path, map_location, torch_sum):
        (model_state, MH_state, summ_dict, MH_dict_orig) = self.cache[load_path]
        model = Model.set_model_state(model_state, self.sig_ref, self.device, torch_sum)
        MH_dict = set_MH_state(model, MH_state)
        return model, MH_dict, summ_dict


def get_rng_seeds_arr(picking_method, picking_args, n_chains):
    """
      Creates the seed numbers array

      :param: picking_method(str): the method for compiling the seed numbers
      :param: picking_args(list):  the parameters used for compiling the seeds.
      :return: rng_seeds(np.ndarray): the array of seeds with the supposed shape
                                      of chain_size.
    """
    if picking_method == 'range':
        rng_seeds = np.arange(*picking_args)
    elif picking_method == 'list':
        rng_seeds = np.array(*picking_args)
    elif picking_method == 'start_step':
        start, step = picking_args
        rng_seeds = start + np.arange(n_chains) * step
    else:
        raise ValueError(f'seeding method {picking_method} not implemented!')
    rng_seeds = rng_seeds.astype(np.int32)
    assert rng_seeds.size == n_chains
    return rng_seeds


def repeat_mcmc_stages(stage_name, init_stage_name, mini_stage_repeats,
                       mini_stage_name_formatter, mini_stage_args):
    """
    The utility function designed to replace "mcmc_repeat" operations
    with multiple "mcmc" operations.

    Arguments:

    :param: stage_name(str): The stage name prefix for all the mini-stages
    :param: init_stage_name(str): The stage from which the mcmc sampling should start.
    :param: mini_stage_repeats(int): The number of mini mcmc stages to repeat,
                                     i.e., mini_stage_repeats == len(output).
    :param: mini_stage_name_formatter(str): string formatter for mini-stage names.
    :param: mini_stage_args(dict): The dictionary of mcmc arguments.
                                   This should not include the past_stage_name.

    Outputs:

    :returns: out(list): The list of repeated mcmc stage dictionaries.

    Example arguments:
      stage_name = "L2"
      init_stage_name = "L0"
      mini_stage_repeats = 2
      mini_stage_name_formatter = "{stage_name}-{mini_stage_idx}"
      mini_stage_args = {"steps": 1000, "after_tune": false,
                         "get_stats": true, "log_acc_rate": true,
                         "log_var_vals": true, "log_interval": 1000}

    Example output:
      out = [{"stage_name": "L2-0",
              "stage_oper": "mcmc",
              "stage_args": {"past_stage_name": "L0", "steps": 1000,
                             "after_tune": false, "get_stats": true,
                             "log_acc_rate": true, "log_var_vals": true,
                             "log_interval": 1000}
              },
              {"stage_name": "L2-1",
               "stage_oper": "mcmc",
               "stage_args": {"past_stage_name": "L2-0", "steps": 1000,
                              "after_tune": false, "get_stats": true,
                              "log_acc_rate": true, "log_var_vals": true,
                              "log_interval": 1000}
               }]
  }
    """
    out = []  # list of mcmc stage dictionaries
    assert 'past_stage_name' not in mini_stage_args, 'you should not passs past_stage_name in mini_stage_args.'
    past_stage_name = init_stage_name
    for mini_stage_idx in range(mini_stage_repeats):
        mini_stage_name = mini_stage_name_formatter.format(stage_name=stage_name,
                                                           mini_stage_idx=mini_stage_idx)
        mini_stage_args = mini_stage_args.copy()
        mini_stage_args['past_stage_name'] = past_stage_name

        mini_stage_dict = dict()
        mini_stage_dict['stage_name'] = mini_stage_name
        mini_stage_dict['stage_oper'] = 'mcmc'
        mini_stage_dict['stage_args'] = mini_stage_args

        out.append(mini_stage_dict)
        # updating past_stage_name for the next repeat
        past_stage_name = mini_stage_name

    return out


def preprocess_curric_config(curric_config_dict):
    """
       Repeats the MCMC stages
    """
    curric_config_dict_post = []
    for stage_dict in curric_config_dict:
        curc_stage_name = stage_dict['stage_name']
        curc_operation = stage_dict['stage_oper']
        curc_stage_args = stage_dict['stage_args']
        if curc_operation == 'mcmc_repeat':
            mcmc_stage_dicts = repeat_mcmc_stages(stage_name=curc_stage_name,
                                                  init_stage_name=curc_stage_args['init_stage_name'],
                                                  mini_stage_repeats=curc_stage_args['mini_stage_repeats'],
                                                  mini_stage_name_formatter=curc_stage_args['mini_stage_name_formatter'],
                                                  mini_stage_args=curc_stage_args['mini_stage_args'])
            curric_config_dict_post = curric_config_dict_post + mcmc_stage_dicts
        else:
            curric_config_dict_post.append(stage_dict)
    return curric_config_dict_post
