import torch
import numpy as np
from collections import OrderedDict

from rng import BatchCudaRNG
from variables import RV
from transforms import Log, StickBreaking, Interval
from distributions import Dirichlet, Normal, Uniform, HalfCauchy, TransformedDistribution


# Model:
class Model:
    def __init__(self, model_param_setting, model_extra_setting):
        self.model_param_setting = model_param_setting
        self.model_extra_setting = model_extra_setting

        # model extra setting
        self.G = model_extra_setting['G']
        self.C = model_extra_setting['C']
        self.N = model_extra_setting['N']
        self.eps = model_extra_setting['eps']
        self.chain_size = model_extra_setting['chain_size']
        self.sampler = model_extra_setting['sampler']
        self.fixed_sigs = model_extra_setting['fixed_sigs']
        self.transformation = model_extra_setting['transformation']
        self.sig_ref = model_extra_setting['sig_ref']

        self.tch_dtype = self.sig_ref.dtype
        self.device = self.sig_ref.device
        self.dim_ext = [1]*len(self.chain_size)  # extending for batch dimensions
        self.w0 = model_extra_setting['w0'].reshape((*self.dim_ext, 1, self.C))
        self.vars = []
        self.var_logps = OrderedDict()
        self.torch_mypi = torch.tensor(np.pi, device=self.device, dtype=self.tch_dtype)

        # model parameters
        # proportions:
        self.W = None

        # alpha for dirichlet:
        self.alpha_distr, hyper_dict = model_param_setting['props_alpha']
        if self.alpha_distr == 'uniform':
            self.alpha_lower = hyper_dict['lower']
            self.alpha_upper = hyper_dict['upper']
        self.alpha = None

        # expression noise sd:
        self.sigma_x_distr, hyper_dict = model_param_setting['x_sigma']
        if self.sigma_x_distr == 'halfcauchy':
            self.beta_x = hyper_dict['beta']
        self.sigma_x = None

        if not self.fixed_sigs:
            # signature noise std
            self.sigma_b_distr, hyper_dict = model_param_setting['sig_sigma']
            if self.sigma_b_distr == 'halfcauchy':
                self.beta_b = hyper_dict['beta']
            self.__dict__['sigma_b'] = None
            self.__dict__['B'] = None

    def get_model_state(self):
        """
        Get the current state of the model which is defined by the current variable values (dict),
        log probability of each variable (dict), state of the random number generator used for sampling (dict),
        as well as model_extra_setting (dict) and model_parameter_setting (dict).
        :return model_state: (dict)
        """
        model_extra_setting = {}
        for k, v in self.model_extra_setting.items():
            if not(k in ['sampler', 'sig_ref']):
                model_extra_setting[k] = v

        sampler = self.model_extra_setting['sampler']
        rng_states = sampler.get_state()  # dictionary of byte-tensors and indices
        var_values = {}
        for var in self.vars:
            var_values[var.name] = var.value

        model_state = dict(model_extra_setting=model_extra_setting, model_param_setting=self.model_param_setting,
                           rng_states=rng_states, var_values=var_values, var_logps=self.var_logps)
        return model_state

    @classmethod
    def set_model_state(cls, model_state, sig_ref, device, torch_sum):
        """
        Creates a new model and sets its state to the given model_state
        :param model_state: (dict) contains model_extra_setting (dict), model_param_setting(dict),
                            value of the variables (dict) and the probability of the variables (dict)
        :param sig_ref: (torch.Tensor) reference cell type signatures used by the model
        :param device: (torch.device) The device to run the model on
        :param torch_sum: Function to perform summation
        :return model: (Model.Model)
        """
        model_extra_setting = model_state['model_extra_setting']
        model_param_setting = model_state['model_param_setting']
        var_values = model_state['var_values']
        model_extra_setting['sig_ref'] = sig_ref

        # create the model sampler from rng states
        sampler = BatchCudaRNG(device, model_extra_setting['chain_size'], var_values['sigma_x'].dtype)
        sampler.set_state(model_state['rng_states'])
        model_extra_setting['sampler'] = sampler

        model = cls(model_param_setting, model_extra_setting)
        model.initialization(init_values=var_values, is_transformed=True,
                             do_prior_init_dict={}, torch_sum=torch_sum)
        for var in model.vars:
            model.var_logps[var.name] = model_state['var_logps'][var.name]
        return model

    @classmethod
    def merge_models(cls, model1, model2, torch_sum):
        """
        Merging two models along the batch dimension (chain dimension)
        1. The sampler of the merged model is the merged of the two samplers
        2. The variable values and log probabilities should be concatenated
           along the chain dimension

        :param model1: (Model.Model) The first model to be merged
        :param model2: (Model.Model) The second model to be merged
        :param torch_sum: Function to perform summation
        :return model: (Model.Model) Merged model
        """

        model_param_setting = model1.model_param_setting.copy()  # or model2.model_param_setting
        model_extra_setting = model1.model_extra_setting.copy()

        # sampler and chain_size should be fixed
        chain_size = (model1.chain_size[0]+model2.chain_size[0],)
        model_extra_setting['chain_size'] = chain_size

        sampler1, sampler2 = model1.sampler, model2.sampler
        sampler = BatchCudaRNG.Merge(sampler1, sampler2)
        model_extra_setting['sampler'] = sampler

        # concat the values
        var_values = {}
        for var in model1.vars:
            var1, var2 = model1.__dict__[var.name].value, model2.__dict__[var.name].value
            var_values[var.name] = torch.cat((var1, var2), dim=0)

        model = cls(model_param_setting, model_extra_setting)
        model.initialization(init_values=var_values, is_transformed=True,
                             do_prior_init_dict={}, torch_sum=torch_sum)
        for var in model1.vars:
            var_logps1, var_logps2 = model1.var_logps[var.name], model2.var_logps[var.name]
            model.var_logps[var.name] = torch.cat((var_logps1, var_logps2), dim=0)
        return model

    @classmethod
    def subsetter_model(cls, model, chain_inds, torch_sum):
        """
        Subset the variables, variable logps (log probabilities), and the sampler
        setting the new chain_size

        :param model: Model instance
        :param chain_inds: (list) chain indices to keep
        :param torch_sum: function to perform summation
        :return model_sub: subset model which has the chains in chain_inds
        """
        chain_size_sub = (len(chain_inds),)
        sampler_sub = BatchCudaRNG.Subset(model.sampler, chain_inds)

        model_param_setting = model.model_param_setting.copy()
        model_extra_setting = model.model_extra_setting.copy()
        model_extra_setting['chain_size'] = chain_size_sub
        model_extra_setting['sampler'] = sampler_sub

        # subsetting the var logps and values
        var_values = {}
        for var in model.vars:
            var_values[var.name] = var.value[chain_inds, :, :]
        model_sub = cls(model_param_setting, model_extra_setting)
        model_sub.initialization(init_values=var_values, is_transformed=True,
                                 do_prior_init_dict={}, torch_sum=torch_sum)
        for var in model.vars:
            assert len(model.var_logps[var.name].shape) == 3, "length of the var logp is not correct"
            model_sub.var_logps[var.name] = model.var_logps[var.name][chain_inds, :, :]
        return model_sub

    def initialization(self, seed=None, init_values=None, is_transformed=False,
                       do_prior_init_dict=None, torch_sum=None):
        """
        Define and initialize the model variables

        :param seed: not used in the current implementation
        :param init_values: (dict) provides the initial value for each variable, the keys are variable names
        :param is_transformed: (boolean) Whether the init values provided are already transformed or not
                              (this is applied towards all variables)
        :param do_prior_init_dict: (dict) determines whether to do prior initialization per variable,
                                    the keys are variable names and the values are boolean (True/False).
        :param torch_sum: function for summation
        :return:
        """

        if init_values is None:
            init_values = {}

        # signature noise and its hyper parameters init
        if not self.fixed_sigs:
            b_distr = Normal(name=f'normal_b', mu=torch.tensor(0., device=self.device, dtype=self.tch_dtype),
                             sigma=torch.tensor(1., device=self.device, dtype=self.tch_dtype), device=self.device,
                             sampler=self.sampler, torch_sum=torch_sum, tch_dtype=self.tch_dtype)
            name = 'B'
            self.__dict__[name] = RV(name=name, type_='free', distribution=b_distr,
                                     size=(*self.chain_size, self.G, self.C))
            self.__dict__[name].init_value(seed=seed, init_value=init_values.get(name, None),
                                           is_transformed=is_transformed,
                                           do_prior_init=do_prior_init_dict.get(name, True))
            self.vars.append(self.__dict__[name])
            self.var_logps[name] = 0.

            if self.sigma_b_distr == 'halfcauchy':
                # signature noise std
                sigma_b_distr = HalfCauchy(name=f'cauchy_sigma_b', beta=self.beta_b, device=self.device,
                                           sampler=self.sampler, torch_sum=torch_sum, tch_dtype=self.tch_dtype)
                transform_cauchy = Log()
                sigma_b_distr_trans = TransformedDistribution(name=f'cauchy_trans_sigma_b',
                                                              dist=sigma_b_distr, transform=transform_cauchy,
                                                              torch_sum=torch_sum)
                name = f'sigma_b'
                self.__dict__[name] = RV(name=name, type_='free',
                                         distribution=sigma_b_distr_trans,
                                         size=(*self.chain_size, 1, self.C), is_transform=True)
                self.__dict__[name].init_value(seed=seed, init_value=init_values.get(name, None),
                                               is_transformed=is_transformed,
                                               do_prior_init=do_prior_init_dict.get(name, True))
                self.vars.append(self.__dict__[name])
                self.var_logps[name] = 0.

        # expression noise init
        if self.sigma_x_distr == 'halfcauchy':
            sigma_x_distr = HalfCauchy(name='cauchy_sigma_x', beta=self.beta_x, device=self.device,
                                       sampler=self.sampler, torch_sum=torch_sum, tch_dtype=self.tch_dtype)
            transform_cauchy = Log()
            sigma_x_distr_trans = TransformedDistribution(name='cauchy_trans_sigma_x',
                                                          dist=sigma_x_distr, transform=transform_cauchy,
                                                          torch_sum=torch_sum)
            name = 'sigma_x'
            self.sigma_x = RV(name='sigma_x', type_='free', distribution=sigma_x_distr_trans,
                              size=(*self.chain_size, 1, 1), is_transform=True)
            self.sigma_x.init_value(seed=seed, init_value=init_values.get('sigma_x', None),
                                    is_transformed=is_transformed,
                                    do_prior_init=do_prior_init_dict.get(name, True))
            self.vars.append(self.sigma_x)
            self.var_logps[name] = 0.

        # as alpha is uniform its transformed should be used
        if self.alpha_distr == 'uniform':
            alpha_distr = Uniform(name='unif_alpha', lower=self.alpha_lower, upper=self.alpha_upper,
                                  device=self.device, sampler=self.sampler, tch_dtype=self.tch_dtype)
            transform_uniform = Interval(a=self.alpha_lower, b=self.alpha_upper)
            alpha_distr_trans = TransformedDistribution(name='unif_trans_alpha', dist=alpha_distr,
                                                        transform=transform_uniform, torch_sum=torch_sum)
            name = 'alpha'
            self.alpha = RV(name='alpha', type_='free', distribution=alpha_distr_trans,
                            size=(*self.chain_size, 1, 1), is_transform=True)
            self.alpha.init_value(seed=seed, init_value=init_values.get('alpha', None),
                                  is_transformed=is_transformed, do_prior_init=do_prior_init_dict.get(name, True))
            self.vars.append(self.alpha)
            self.var_logps[name] = 0.

        # weights with dirichlet distribution:
        W_distr = Dirichlet(name='dirch_w', a=self.w0, device=self.device,
                            sampler=self.sampler, torch_sum=torch_sum, tch_dtype=self.tch_dtype)
        transform_dirch = StickBreaking(eps=self.eps, device=self.device, torch_sum=torch_sum)
        W_distr_trans = TransformedDistribution(name='dirch_trans_w', dist=W_distr,
                                                transform=transform_dirch, torch_sum=torch_sum)
        name = 'W'
        self.W = RV(name='W', type_='free', distribution=W_distr_trans,
                    size=(*self.chain_size, self.C, self.N),
                    dependency=['alpha'], is_transform=True)
        point_cond = {}
        if do_prior_init_dict.get(name, True):
            assert not(self.alpha.value is None), "prior initialization for W. Proper alpha intialization is needed"
            a_inv = self.alpha.backward_var(self.alpha.value)  # used when doing prior init for W
            point_cond = {'a': a_inv*self.w0}  # a_inv (chain_size, 1, 1), w0 is (1, 1, C)
        self.W.init_value(point=point_cond, seed=seed, init_value=init_values.get('W', None),
                          is_transformed=is_transformed, do_prior_init=do_prior_init_dict.get(name, True))
        self.vars.append(self.W)
        self.var_logps[name] = 0.

    def forward(self, point=None):
        """
        Do a forward pass to compute SW (multiplication of signatures and proportions)
        :param point: (dict) The point at which forward pass is done. Keys are variable names
                      and the values are variable values. if not provided the model current
                      value of the variable/parameter is used.
        :return mu_x: (torch.Tensor) mu_x= SW
        """

        sigma_b, B = None, None
        if point:  # when an update is provided
            W = self.W.backward_var(point['W'])  # current value provided for W
            if not self.fixed_sigs:
                # extracting sigma_b for each cell type
                sigma_b = self.__dict__['sigma_b'].backward_var(point['sigma_b'])
                B = point['B']
        else:
            W = self.W.backward_var()  # if W is not transformed then self.W.value is returned
            if not self.fixed_sigs:
                sigma_b = self.__dict__['sigma_b'].backward_var()
                B = self.__dict__['B']

        if not self.fixed_sigs:
            u = self.sig_ref  # (G, C)
            u = u.reshape((*self.dim_ext, self.G, self.C))  # adding the chain dimension
            cv = u * (sigma_b**2)  # sigma_b is (self.chain_size, 1, self.C) and u is (1, G, C)
            cv = torch.sqrt(torch.abs(cv))
            S = u + (B*cv)  # element-wise multiplication
            if self.transformation == "log":
                S = torch.exp(S)
        else:
            S = self.sig_ref  # (G, C)
            S = S.reshape((*self.dim_ext, self.G, self.C))  # (1, G, C)
        mu_x = S@W
        return mu_x

    def LL(self, observed, mu, sd, torch_sum):
        """
        Compute the loglikelihood of the data (X ~ N(SW, sigma_x))
        :param observed: (torch.Tensor) size (G, N)
        :param mu: (torch.Tensor) size (chain_size, G, N)
        :param sd: (torch.Tensor) size (chain_size, 1, 1)
        :param torch_sum: function to perform summation
        :return LL: (torch.Tensor) loglikelihood size (chain_size, 1, 1)
        """

        G, N = observed.shape[-2:]
        observed = observed.reshape((*self.dim_ext, G, N))  # (1, G, N)
        SE = 0.5*torch.square((observed - mu)/sd)  # size= (chain_size, G, N)
        LL = -G*N*(torch.log(sd) + 0.5*torch.log(2.*self.torch_mypi)) - \
            torch_sum(SE, dim=(-2, -1), keepdim=True)  # (chain_size, 1, 1)
        return LL  # (chain_size, 1, 1)

    def logp(self, observed, point=None, torch_sum=None):
        """
        Compute the posterior
        The point could contain the value for some or all of the variables used for
        log probability calculation. in case some variables are missing their current value will be used
        This function iterates over all the variables and computes the log probability for them.

        :param observed: (torch.Tensor) bulk expression profiles
        :param point: (dict) The point at which posterior is computed. The keys are variable names.
                      and the values are variable values. If not provided the current parameter values
                      in the model will be used.
        :param torch_sum: Function to perform summation
        :return: (torch.Tensor) posterior is returned
        """

        logp_priors = 0.
        fw_point = {}
        fw_var_names = ['W']

        if not self.fixed_sigs:
            # sigma_b and signature bias and scale
            fw_var_names.append('sigma_b')
            fw_var_names.append('B')

        for var in self.vars:
            # if the point is not available then compute logp for the current values
            if point and var.name in point.keys():
                var_value = point[var.name]
                if var.name in fw_var_names:
                    fw_point[var.name] = point[var.name]
            else:
                var_value = var.value
                if var.name in fw_var_names:
                    fw_point[var.name] = var_value

            point_cond = None
            if var.name == 'W':
                point_cond = {}
                if point and 'alpha' in point.keys():
                    a_inv = self.alpha.backward_var(point['alpha'])
                    point_cond['a'] = a_inv*self.w0
                else:
                    a_inv = self.alpha.backward_var()
                    point_cond['a'] = a_inv*self.w0
            logp_priors = logp_priors + var.logp(value=var_value, point=point_cond)

        mu_x = self.forward(point=fw_point)  # this should be fixed if later you plan to provide partial info
        # in the point dict

        if point and 'sigma_x' in point.keys():
            sd_x = self.sigma_x.backward_var(point['sigma_x'])
        else:
            sd_x = self.sigma_x.backward_var()
        if torch.any(sd_x < 0):
            raise ValueError("negative standard deviation encountered")

        LL = self.LL(observed=observed, mu=mu_x, sd=sd_x, torch_sum=torch_sum)

        return logp_priors + LL

    def logp_modif(self, observed, point=None, overwrite=False,
                   in_update=False, acc_vars=None, torch_sum=None):
        """
        An efficient way to compute the posterior
        The posterior is the sum of log likelihood and log prior for all the variables
        If only one variable has changed value there is no need to redo the log prior
        calculation for all the variables. The change of log prior for the variable with
        changed value should be added to the current sum of log priors over the variables
        which is then used for posterior calculation.

        :param observed: (torch.Tensor) bulk expression profiles
        :param point: (dict) The point at which posterior is computed. If the point is None
                      then log prior probability of all model variables are computed and summed.
                      Also, the current value of model variables is used for prior calculation.
        :param overwrite: (boolean) the logp of the variable is updated with the computed logp,
                          in the model.
        :param in_update: (boolean) when the point contains at least one variable
                          and the deviation from the current posterior (contributed by
                          variable prior probability) should be computed.
        :param acc_vars: (set to None for now) -- do not use this
        :param torch_sum: function used for summation
        :return: (torch.Tensor) posterior is returned
        """

        fw_point = {}
        fw_var_names = ['W']

        if not self.fixed_sigs:
            fw_var_names.append('sigma_b')
            fw_var_names.append('B')

        if point is None:
            logp_priors = 0.
        elif in_update:
            logp_priors = 0
            for _, (kkk, vvv) in enumerate(self.var_logps.items()):
                logp_priors = logp_priors + vvv
            assert tuple(logp_priors.shape[:-2]) == self.chain_size, f'{logp_priors.shape}'
        else:
            raise Exception('not implemented error!')

        for var in self.vars:
            # if the point is not available then compute logp for the current values
            if point and (var.name in point.keys()):
                var_value = point[var.name]
                if var.name in fw_var_names:
                    fw_point[var.name] = point[var.name]
            else:
                var_value = var.value
                if var.name in fw_var_names:
                    fw_point[var.name] = var_value

            point_cond = None
            if var.name == 'W':
                point_cond = {}
                if point and ('alpha' in point.keys()):
                    a_inv = self.alpha.backward_var(point['alpha'])
                    point_cond['a'] = a_inv*self.w0  # self.w0 has dimension (1,1,C)
                else:
                    a_inv = self.alpha.backward_var()
                    point_cond['a'] = a_inv*self.w0

            if point is None:  # if no point is provided simply compute the logp of all variables
                if acc_vars is None:
                    var_logp = var.logp(value=var_value, point=point_cond)
                    logp_priors = logp_priors + var_logp
                    if overwrite:
                        self.var_logps[var.name] = var_logp
                else:
                    if (var.name in acc_vars.keys() and acc_vars[var.name]) or (not(var.name in acc_vars.keys())):
                        var_logp = var.logp(value=var_value, point=point_cond)
                        if overwrite:  # when you intialized variables you should call model.logp with overwrite= True
                            self.var_logps[var.name] = var_logp
                    elif (var.name in acc_vars.keys() and not acc_vars[var.name]):
                        if var.name == 'W' and acc_vars['alpha']:
                            var_logp = var.logp(value=var_value, point=point_cond)
                            if overwrite:
                                self.var_logps[var.name] = var_logp
                        else:
                            var_logp = self.var_logps[var.name]
                    logp_priors = logp_priors + var_logp

            elif in_update:  # in updating at least one variable value has changed
                if var.name in point.keys():  # only do the prior calculations for the variable in the point
                    curr_prior = self.var_logps[var.name]
                    new_prior = var.logp(value=var_value, point=point_cond)
                    diff_prior = new_prior - curr_prior
                    logp_priors = logp_priors + diff_prior
                    if overwrite:
                        self.var_logps[var.name] = new_prior

                    if var.name == 'alpha' and not 'W' in point.keys():
                        a_inv = self.alpha.backward_var(point['alpha'])
                        point_cond = dict()
                        point_cond['a'] = a_inv*self.w0
                        curr_prior = self.var_logps['W']
                        new_prior = self.__dict__['W'].logp(value=self.__dict__['W'].value, point=point_cond)
                        diff_prior = new_prior - curr_prior
                        logp_priors = logp_priors + diff_prior
                        if overwrite:
                            self.var_logps['W'] = new_prior
            else:
                raise Exception('not implemented error!')

        mu_x = self.forward(point=fw_point)
        # in the point dict
        if point and 'sigma_x' in point.keys():
            sd_x = self.sigma_x.backward_var(point['sigma_x'])
        else:
            sd_x = self.sigma_x.backward_var()
        if torch.any(sd_x < 0):
            raise ValueError("negative standard deviation encountered")

        LL = self.LL(observed=observed, mu=mu_x, sd=sd_x, torch_sum=torch_sum)
        return logp_priors + LL
