import torch
import time
import numpy as np
from collections import defaultdict
from proposals import NormalProposal, MultivariateNormalProposal
from summary import compute_summary_multichain


class Metropolis:
    name = 'metropolis'

    generates_stats = True
    stats_dtypes = [{
        'accept': torch.float64,
        'accepted': np.bool,
        'tune': np.bool,
        'scaling': torch.float64,
    }]

    def __init__(self, var=None, S=None, proposal_dist=None, scaling=1.,
                 tune=True, tune_interval=100, temperature=1.,
                 device='cpu', chain_size=None, sampler=None,
                 tch_dtype=torch.float32, divide_B=True):
        """

        :param var: (variables.RV) The variable object
        :param S: leave it at None
        :param proposal_dist:
        :param scaling: (torch.Tensor)
        :param tune: (boolean)
        :param tune_interval: (int) the interval to perform tuning, i.e. adjusting the scale used for
                              parameter updates
        :param temperature: (float) leave it at default 1.
        :param device: (torch.Tensor) The device to perform computations on
        :param chain_size: (tuple) This is in tuple format in case more batch dimensions need to be added
                            currently the tuple it is (chain_numbers, ) e.g. (150, )
        :param sampler: (rng.BatchCudaRNG) The sampler used for proposing new variable values and
                        sampling from uniform for the acceptance of the update
        :param tch_dtype: dtype of the torch Tensor e.g. torch.float32
        :param divide_B: (boolean) Whether signature noise std is cell type specific
                         This is applied only to the "B" variable which is for the signature noise
        """

        if S is None:
            print(var.name, var.size_())
            S = torch.ones(var.size_(), device=device, dtype=tch_dtype)
            # S = np.ones(sum(v.size_() for v in var))

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.dim() == 1:
            self.proposal_dist = NormalProposal(S, sampler)
        elif S.dim() == 2:
            self.proposal_dist = MultivariateNormalProposal(S)  # not used therefore not fixed
        else:
            raise ValueError("Invalid rank for variance: %s" % S.dim())

        self.var_size = None  # for sigma_x, sigma_b that are scalar
        if var.size:
            self.var_size = var.value.shape  # or =var.size

        self.device = device
        self.temperature = temperature
        self.var_name = var.name
        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.tch_dtype = tch_dtype
        self.divide_B = divide_B
        self.sampler = sampler
        self.use_batch_rng = sampler.is_batch
        # print(f"batch_rng in MH is {self.use_batch_rng}") # for check only. remove later

        assert not(chain_size is None), "chain size must be provided"
        self.chain_size = chain_size

        if self.var_name == 'B' and divide_B:
            # divide_B means that signature noise std is cell type specific
            C = self.var_size[-1]
            self.accepted = 0. if (self.var_size is None) \
                else torch.zeros((*self.var_size[:-2], 1, C),
                                 device=self.device, dtype=self.tch_dtype)
        else:
            self.accepted = 0. if (self.var_size is None) \
                else torch.zeros((*self.var_size[:-2], 1, 1),
                                 device=self.device, dtype=self.tch_dtype)  # (chain_size, 1, 1) this changes for B

        self._untuned_settings = dict(
            scaling=self.scaling,
            steps_until_tune=tune_interval,
            accepted=self.accepted
        )

    def reset_accepted(self):
        if self.var_name == 'B' and self.divide_B:
            C = self.var_size[-1]
            self.accepted = 0. if (self.var_size is None) \
                else torch.zeros((*self.var_size[:-2], 1, C),
                                 device=self.device, dtype=self.tch_dtype)  # (chain_size, 1, 1) this changes for B
        else:
            self.accepted = 0. if (self.var_size is None) \
                else torch.zeros((*self.var_size[:-2], 1, 1),
                                 device=self.device, dtype=self.tch_dtype)  # (chain_size, 1, 1) this changes for B

    def reset_tuning(self):
        # not implemented
        return

    def update_temperature(self):
        self.temperature = min(self.temperature*2., 1.)

    def astep(self, observed, curr_val, model, curr_post=None, prev_mode=False, torch_sum=None):
        """
        Propose a new update for the variable. Accept or reject the update.
        :param observed: (torch.Tensor) Bulk expression profiles
        :param curr_val: (torch.Tensor) Current value of the parameter
        :param model: (Model.Model) The model object
        :param curr_post: (torch.Tensor) The current posterior of the model
        :param prev_mode: Set to False
        :param torch_sum: Function to perform summation
        :return: (tuple) contains the final value of the parameter and the stats associated with it
        """
        if (not self.steps_until_tune) and self.tune:
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.reset_accepted()

        delta = self.proposal_dist()
        if self.var_size:
            delta = delta.reshape(self.var_size)
        delta = delta * self.scaling

        if self.var_name == 'B' and self.divide_B:
            accepted_lst = []
            G, C = self.var_size[-2:]
            final_val_ = 0.
            for i in range(C):
                mask_arr = torch.zeros((*self.chain_size, G, C), device=self.device, dtype=self.tch_dtype)
                mask_arr[..., :, i] = 1.
                delta_ = delta*mask_arr
                new_val = curr_val + delta_
                accept = decision(model=model, observed=observed,
                                  var_name=self.var_name, new_val=new_val,
                                  curr_val=curr_val, curr_post=curr_post,
                                  prev_mode=prev_mode, temperature=self.temperature, torch_sum=torch_sum)
                final_val, accepted = update(accept, new_val, curr_val, device=self.device, tch_dtype=self.tch_dtype,
                                             use_batch_rng=self.use_batch_rng, sampler=self.sampler)
                final_val_ = final_val_ + final_val*mask_arr
                accepted_lst.append(accepted)
            final_val = final_val_
            accepted = torch.cat(accepted_lst, dim=-1)  # (chain_size, 1, C)
            assert accepted.shape == (*self.chain_size, 1, C), "size mismatch in MH"
        else:
            new_val = curr_val + delta
            accept = decision(model=model, observed=observed,
                              var_name=self.var_name, new_val=new_val,
                              curr_val=curr_val, curr_post=curr_post,
                              prev_mode=prev_mode, temperature=self.temperature, torch_sum=torch_sum)
            final_val, accepted = update(accept, new_val, curr_val, device=self.device, tch_dtype=self.tch_dtype,
                                         use_batch_rng=self.use_batch_rng, sampler=self.sampler)

        # maybe later you could remove this
        if (not final_val.dim() == 0) and final_val.numel() == 1:  # like torch.tensor([1]) where dim is 1 but there is one element only
            final_val = final_val[0]

        self.accepted += accepted  # this is for keeping track of the acceptance rate
        self.steps_until_tune -= 1

        if (not self.steps_until_tune) and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'accept': accept,
            'accepted': accepted
        }

        return final_val, [stats]


def tune(scale, acc_rate):
    """
    The logic of this function is adapted from pymc3, https://github.com/pymc-devs/pymc
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    mask1, mask2, mask3 = acc_rate < 0.001, acc_rate < 0.05, acc_rate < 0.2
    mask4, mask5, mask6 = acc_rate > 0.95, acc_rate > 0.75, acc_rate > 0.5
    mask_neutral = (~mask3)*(~mask6)  # scale stays the same

    mask1_final = mask1
    mask2_final = (~mask1)*mask2
    mask3_final = (~mask2)*mask3

    mask4_final = mask4
    mask5_final = (~mask4)*mask5
    mask6_final = (~mask5)*mask6

    scale1, scale2, scale3 = scale * 0.1, scale * 0.5, scale * 0.9
    scale4, scale5, scale6 = scale * 10.0, scale * 2.0, scale * 1.1

    return scale1*mask1_final + scale2*mask2_final + scale3*mask3_final + \
        scale4*mask4_final + scale5*mask5_final + scale6*mask6_final + scale*mask_neutral


def decision(model, observed, var_name, new_val, curr_val, curr_post=None,
             prev_mode=False, temperature=1., torch_sum=None):
    new_point = {var_name: new_val}
    if prev_mode:
        new_posterior = model.logp(observed=observed, point=new_point, torch_sum=torch_sum)
    else:
        new_posterior = model.logp_modif(observed=observed, point=new_point,
                                         in_update=True, overwrite=False, torch_sum=torch_sum)

    if (curr_post is None):
        curr_posterior = model.logp(observed=observed, point={var_name: curr_val}, torch_sum=torch_sum)
    else:
        curr_posterior = curr_post
    accept = new_posterior - curr_posterior  # assert size=(chain_size, 1, 1)

    # in case simulated annealing is used
    mask = accept < 0  # accept size (chain_size, 1, 1)
    return accept*temperature*mask + accept*(~mask)


def update(accept, new_val, curr_val, device, tch_dtype, use_batch_rng=True, sampler=None):
    if use_batch_rng:
        unif_sample_ = sampler.uniform(sample_shape=accept.shape)
    else:
        unif_ = torch.distributions.uniform.Uniform(low=torch.tensor(0., device=device, dtype=tch_dtype),
                                                    high=torch.tensor(1., device=device, dtype=tch_dtype))
        unif_sample_ = unif_.sample(sample_shape=accept.shape)
    unif_sample = torch.log(unif_sample_)
    cond1, cond2 = torch.isfinite(accept), unif_sample < accept
    mask = cond1*cond2
    final_val = new_val*mask + curr_val*(~mask)  # (chain_size, 1, 1) or (chain_size, 1, C) for B variable
    flag = torch.ones(accept.shape, dtype=torch.bool, device=device)
    final_flag = flag*mask
    return final_val, final_flag


# get MH state after getting model state
def get_MH_state(MH_dict):
    """
    Get MH state after getting model state

    :param MH_dict: (dict) MH (MH.Metropolis) for each variable of the model
    :return MH_state: (dict) state of the Metropolis Hasting
    """
    # no need to get the sampler state because it is already handled in getting model state
    scaling_dict, accepted_dict, steps_until_tune_dict = {}, {}, {}
    tune, tune_interval = None, None
    for var_name, var_MH in MH_dict.items():
        scaling_dict[var_name] = var_MH.scaling
        accepted_dict[var_name] = var_MH.accepted
        steps_until_tune_dict[var_name] = var_MH.steps_until_tune

        tune, tune_interval = var_MH.tune, var_MH.tune_interval
    MH_state = dict(scaling_dict=scaling_dict, tune=tune,
                    tune_interval=tune_interval,
                    divide_B=MH_dict['B'].divide_B,
                    accepted_dict=accepted_dict,
                    steps_until_tune_dict=steps_until_tune_dict)  # divide_B exists for all vars though
    return MH_state


# set MH state after setting model state
def set_MH_state(model, MH_state):
    """
    Create Metropolis Hasting object for each variable using the given MH_state and model
    """
    scaling_dict, divide_B = MH_state['scaling_dict'], MH_state['divide_B']
    tune_interval, tune = MH_state['tune_interval'], MH_state['tune']
    accepted_dict = MH_state.get('accepted_dict', None)
    steps_until_tune_dict = MH_state.get('steps_until_tune_dict', None)

    MH_dict = {}
    for var in model.vars:
        scaling = scaling_dict[var.name]  # retrieve the scalings
        temperature = torch.tensor(1., device=model.device, dtype=model.tch_dtype)
        MH = Metropolis(var=var, scaling=scaling, tune_interval=tune_interval,
                        tune=tune, temperature=temperature, device=model.device,
                        sampler=model.sampler, chain_size=model.chain_size, tch_dtype=model.tch_dtype,
                        divide_B=divide_B)
        if not(accepted_dict is None):
            MH.accepted = accepted_dict[var.name]
        if not(steps_until_tune_dict is None):
            MH.steps_until_tune = steps_until_tune_dict[var.name]
        MH_dict[var.name] = MH
    return MH_dict


def merge_MH(MH1_dict, MH2_dict, model):
    """
    For each variable merge the two MH objects (MH.Metropolis) after merging the
    two models associated with them

    :param MH1_dict: (dict) MH dict associated with the variables of the first model
    :param MH2_dict: (dict) MH dict associated with the variables of the second model
    :param model: (Model.Model) a merged model along the chain dimension
    :return MH_dict: (dict) merged MH_dict (basically the scalings are merged)
    """

    MH_dict = {}
    for var in model.vars:
        assert MH1_dict[var.name].tune == MH2_dict[var.name].tune, 'tune state does not match'
        scaling1, scaling2 = MH1_dict[var.name].scaling, MH2_dict[var.name].scaling  # retrieve the scalings
        scaling = torch.cat((scaling1, scaling2), dim=0)
        temperature = torch.tensor(1., device=model.device, dtype=model.tch_dtype)
        MH = Metropolis(var=var, scaling=scaling, tune_interval=MH1_dict[var.name].tune_interval,
                        tune=MH1_dict[var.name].tune, temperature=temperature, device=model.device,
                        sampler=model.sampler, chain_size=model.chain_size, tch_dtype=model.tch_dtype,
                        divide_B=MH1_dict['B'].divide_B)
        MH_dict[var.name] = MH
    return MH_dict


# subset the MH after subsetting the model
def subsetter_MH(MH_dict, model_sub, chain_inds):
    """
    For each variable subset the MH after subsetting the model

    :param MH_dict: (dict) Contains MH for the variables
    :param model_sub: (Model.Model) The subset model
    :param chain_inds: The chain indices to subset the MH for each variable
                       basically only the scalings are subset to chain_inds
    :return MH_dict_sub: (dict) MH_dict with subset MH for each variable
    """

    MH_dict_sub = {}
    for var in model_sub.vars:
        scaling = MH_dict[var.name].scaling[chain_inds, :, :]  # retrieve the scalings
        temperature = torch.tensor(1., device=model_sub.device, dtype=model_sub.tch_dtype)
        MH = Metropolis(var=var, scaling=scaling, tune_interval=MH_dict[var.name].tune_interval,
                        tune=MH_dict[var.name].tune, temperature=temperature, device=model_sub.device,
                        sampler=model_sub.sampler, chain_size=model_sub.chain_size, tch_dtype=model_sub.tch_dtype,
                        divide_B=MH_dict['B'].divide_B)
        MH_dict_sub[var.name] = MH
    return MH_dict_sub


def random_walk(model, MH_dict, mixtures, steps, after_tune, org_var_dict,
                get_stats, log_acc_rate, log_var_vals, torch_sum, logger, log_interval):
    """

    :param model: (Model.Model) The model
    :param MH_dict: (dict) Contains Metropolis Hasting (MH) object for each variable
    :param mixtures: (torch tensor) bulk expression profiles
    :param steps: (int) The number of MH sampling steps (steps of the random walk)
    :param after_tune: (boolean) determines whether MH is in tuning period or not
    :param org_var_dict: (dict) contains reference signature, original signature and proportions if available
    :param get_stats: (boolean) get the stat summary to keep track of proper model and random walk setup
                     (use for method development or in cases where ground truth for proportions or signatures exist)
    :param log_acc_rate: (boolean) log the acceptance rate so far
    :param log_var_vals: (boolean) log the variable values
    :param torch_sum: function to perform summation
    :param logger: (IO.MHlogger) logger object to log the stats
    :param log_interval: the interval of logging variable values, acceptance rate and getting the stats
    :return: (dict) summ_dict_out, contains the running sum of the samples and their acceptance for each variable
             the summation is over the accepted samples
    """

    # posterior with the initialized model variables
    current_posterior = model.logp_modif(observed=mixtures, point=None, in_update=False,
                                         overwrite=True, torch_sum=torch_sum, acc_vars=None)
    logger(f'current posterior (initialized): {current_posterior}\n')
    acceptance_rate_dict = defaultdict(list)
    summ_dict, accept_ctr = {}, {}  # keeping the sum of accepted samples and the count of them
    start_time = time.time()
    with torch.no_grad():
        for step in range(steps):
            for var in model.vars:
                MH_dict[var.name].tune = not(after_tune)

            samples_dict, acc_vars = {}, {}
            accepted_lst = 0.

            # Perform a Block-wise proposal for variables
            # Propose a new update for one variable at a time then asses to keep or reject it
            # Reference: https://theclevermachine.wordpress.com/2012/11/04/mcmc-multivariate-\
            # distributions-block-wise-component-wise-updates/

            for var in model.vars:
                MH = MH_dict[var.name]
                final_val, [stats_] = MH.astep(observed=mixtures, curr_val=var.value,
                                               model=model, curr_post=current_posterior,
                                               prev_mode=False, torch_sum=torch_sum)
                samples_dict[var.name] = (final_val, stats_)
                if var.name != 'B':
                    # (chain_size, 1, 1) this is a boolean sum so any var with an accepted update leads to true accept
                    accepted_lst = accepted_lst + stats_['accepted']
                else:
                    accepted_lst = accepted_lst + torch_sum(stats_['accepted'], dim=-1, keepdim=True)
                acceptance_rate_dict[var.name].append(stats_['accepted'])
                acc_vars[var.name] = stats_['accepted']

            # Update the variable values after proposal assessment for all variables is finished
            for var in model.vars:
                (final_val, stats_) = samples_dict[var.name]
                var.value = final_val  # size is (chain_size, ...)

            current_posterior = model.logp_modif(observed=mixtures, point=None,
                                                 in_update=False, overwrite=True,
                                                 acc_vars=None, torch_sum=torch_sum)
            # compute the sample summaries (sum of accepted samples)
            sample_accepted = accepted_lst.bool()
            for var in model.vars:
                var_value_untrans = var.backward_var(samples_dict[var.name][0])
                var_value_untrans = var_value_untrans*sample_accepted
                if not(var.name in summ_dict.keys()):
                    summ_dict[var.name] = var_value_untrans
                    accept_ctr[var.name] = 0.
                    accept_ctr[var.name] = accept_ctr[var.name] + sample_accepted
                else:
                    summ_dict[var.name] = summ_dict[var.name] + var_value_untrans
                    accept_ctr[var.name] = accept_ctr[var.name] + sample_accepted

            # getting the variable values, for debugging purposes
            if log_var_vals and (step+1) % log_interval == 0:
                logger('getting variable values ...\n')
                sigma_x_rep = model.sigma_x.distribution.transform.backward(model.sigma_x.value).reshape(-1)
                alpha_rep = model.alpha.distribution.transform.backward(model.alpha.value).reshape(-1)
                if not model.fixed_sigs:
                    sigma_b_rep = np.array(model.sigma_b.distribution.transform.backward(model.sigma_b.value).cpu()).reshape((*model.chain_size, model.C))
                    logger(f'step {step}\n,posterior \n{current_posterior.squeeze()}\n', stdout=True)
                    logger(f'sigma_x\n{sigma_x_rep}\n,alpha\n{alpha_rep}\n,sigma_b\n{sigma_b_rep}\n')
                else:
                    logger(f'step {step},\ncurrent_posterior\n{current_posterior},\nsigma_x\n{sigma_x_rep},\nalpha\n{alpha_rep}\n')
                logger.flush()

            # getting the stat summary
            if get_stats and (step+1) % log_interval == 0:
                logger('getting the stat summaries ...\n')
                avg_speed = 1000. * (time.time() - start_time) / (step+1)
                logger(f'step {step} (%.2f ms/step)\n' %(avg_speed), stdout=True)
                stat_tuple = compute_summary_multichain(model, mixtures, summ_dict,
                                                        accept_ctr, org_var_dict, torch_sum=torch_sum)
                mean_dict, MSE_dict, _ = stat_tuple
                # logging the stat summaries
                X_n, W_n, S_n = 'X', 'W', 'S'
                W_corr_n, S_corr_n = 'W_corr_CT', 'S_corr_CT'
                if not model.fixed_sigs:
                    logger(f'X_MSE\n{MSE_dict[X_n].squeeze()}\n')
                    if not(org_var_dict["props"] is None):
                        # proportion stats exist
                        logger(f'W_MSE\n{MSE_dict[W_n].squeeze()}\n')
                        logger(f'W_corr\n{MSE_dict[W_corr_n]}\n', stdout=True)
                    if not(org_var_dict["sig_org"] is None):
                        # signature stats exist
                        logger(f'S_MSE\n{MSE_dict[S_n].squeeze()}\n')
                        logger(f'S_corr\n{MSE_dict[S_corr_n]}\n', stdout=True)
                else:
                    if not(org_var_dict["props"] is None):
                        logger(f'W_MSE\n{MSE_dict[W_n].squeeze()}\n')
                        logger(f'W_corr\n{MSE_dict[W_corr_n]}\n', stdout=True)
                logger.flush()
                summ_dict, accept_ctr = {}, {}

            # keeping track of the acceptance rate in case tunning is true
            if log_acc_rate and (step+1) % log_interval == 0:
                logger('acceptance rate\n', stdout=True)
                divide_B = MH_dict['B'].divide_B
                for var in model.vars:
                    dim_cat = -2 if (var.name == 'B' and divide_B) else -1
                    acceptance_rate_sofar = torch.cat(acceptance_rate_dict[var.name], dim=dim_cat)
                    acceptance_rate_sofar = torch_sum(acceptance_rate_sofar, dim=dim_cat,
                                                      keepdim=True)/len(acceptance_rate_dict[var.name])  # (chain_size, 1, 1)
                    acceptance_rate_sofar = acceptance_rate_sofar.reshape((*model.chain_size, model.C)) \
                        if (var.name == 'B' and divide_B) \
                        else acceptance_rate_sofar.reshape(-1)
                    logger(f'{var.name},\n{acceptance_rate_sofar}\n', stdout=True)
                logger.flush()

    # build the appropriate summary to save (tuples should be saved)
    summ_dict_out = {}
    for var_name in summ_dict.keys():
        summ_dict_out[var_name] = (summ_dict[var_name], accept_ctr[var_name])
    return summ_dict_out
