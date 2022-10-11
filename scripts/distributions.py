import torch
import numpy as np
from transforms import StickBreaking


class Distribution:
    def __init__(self, name, transform=None, device='cpu',
                 sampler=None, torch_sum=None, tch_dtype=None):
        self.device = device
        self.name = name
        self.transform = transform
        self.sampler = sampler
        self.torch_sum = torch_sum
        self.tch_dtype = tch_dtype

        if True:
            self.POSINF = torch.finfo(self.tch_dtype).max
            self.NEGINF = torch.finfo(self.tch_dtype).min
        else:
            self.POSINF = float('Inf')
            self.NEGINF = -float('Inf')

    def generate_samples(self):
        raise NotImplementedError

    def logp(self, value):
        raise NotImplementedError


class Dirichlet(Distribution):
    # a has variable type
    def __init__(self, name, a, transform=StickBreaking(eps=1e-9),
                 device='cpu', sampler=None, torch_sum=None, tch_dtype=None):
        super().__init__(name=name, transform=transform, device=device,
                         sampler=sampler, torch_sum=torch_sum, tch_dtype=tch_dtype)
        self.a = a
        self.mean = a/self.torch_sum(a, dim=-1, keepdim=True)  # a size is (chain_size, 1, C)

    def get_mean(self):
        return self.mean

    def generate_samples(self, size=None, point=None, seed=None):  # not implemented in 3D version
        a = self.a  # (chain_size, 1, C)
        if point:
            try:
                a = point['a']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        # cannot provide random state to the generator
        assert (a.shape[-1] == size[-2]), "number of CTs in a do not match the CTs in the W"
        C, N = size[-2:]  # shape of size (chain_size, C, N)
        if not self.sampler.is_batch:
            a_expanded = a.expand((*a.shape[:-2], N, C))  # (chain_size, N, C)
            gen = torch.distributions.dirichlet.Dirichlet(concentration=a_expanded)
            samples = gen.sample(sample_shape=[1])  # samples shape (1, chain_size, N, C)
            samples = samples.squeeze()  # (chain_size, N, C)
        else:
            chain_size_tuple = a.shape[:-2]  # (chain_size,)
            a_2d = a.reshape(*chain_size_tuple, C)  # (chain_size, C)
            gen_list = [torch.distributions.dirichlet.Dirichlet(concentration=a_1d) for a_1d in a_2d]
            samples = self.sampler.dirichlet(gen_list, (*chain_size_tuple, N))  # (chain_size, N, C)
            assert samples.shape == (*chain_size_tuple, N, C)

        return torch.transpose(samples, dim0=-2, dim1=-1)  # (chain_size, C, N)

    def logp(self, value, point=None):
        a = self.a  # (chain_size, 1, C)
        if point:
            try:
                a = point['a']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        cond1 = torch.all(torch.all(value >= 0., dim=-1, keepdim=True), dim=-2, keepdim=True)  # (chain_size, 1, 1)
        cond2 = torch.all(torch.all(value <= 1., dim=-1, keepdim=True), dim=-2, keepdim=True)  # (chain_size, 1, 1)
        cond3 = torch.all(torch.all(a > 0, dim=-1, keepdim=True), dim=-2, keepdim=True)  # (chain_size, 1, 1)
        mask = cond1*cond2*cond3  # (chain_size, 1, 1)

        if value.dim() == 1:
            value = value.reshape(-1, 1)

        if a.shape[-1] != value.shape[-2]:  # value size is (chain_size, C, N)
            raise ValueError("dimension mismatch between a and value")

        sampleSize = value.shape[-1]  # used to be value.shape[1] , should be N  value (chain_size, C, N)
        a_col = torch.transpose(a, dim0=-2, dim1=-1)  # (chain_size, C, 1) #used to be a.reshape((-1,1)) # C*1
        value_powered = torch.log(torch.pow(value, a_col-1))

        B_a = self.torch_sum(torch.lgamma(a), dim=(-2, -1), keepdim=True) \
            - torch.lgamma(self.torch_sum(a, dim=(-2, -1), keepdim=True))  # (chain_size, 1, 1)
        logp = self.torch_sum(value_powered, dim=(-2, -1), keepdim=True) - sampleSize*B_a  # (chain_size, 1, 1)
        logp[torch.logical_not(mask)] = self.NEGINF  # -float('Inf')
        return logp


class Normal(Distribution):
    def __init__(self, name, mu=torch.tensor(0.), sigma=None,
                 device='cpu', sampler=None, torch_sum=None, tch_dtype=None):
        super().__init__(name=name, device=device, sampler=sampler,
                         torch_sum=torch_sum, tch_dtype=tch_dtype)
        self.sigma = sigma
        self.tau = sigma**-2

        self.mu = self.mean = self.median = self.mode = mu  # mu should be float with enough presicion
        self.variance = 1.0/self.tau

    def generate_samples(self, size=None, point=None, seed=None):  # size = (samplesize, mu size)
        mu = self.mu
        scale = self.tau**-0.5
        if point:
            try:
                mu = point['mu']
                scale = point['tau']**-0.5
                sigma = point['sigma']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        gen = torch.distributions.normal.Normal(loc=mu, scale=scale)
        samples = self.sampler(gen, sample_shape=size)  # or torch.Size(size)
        return samples

    def logp(self, value, point=None):
        '''
            value size (chain_size, G, C)
        '''

        sigma = self.sigma
        tau = self.tau
        mu = self.mu

        if point:
            try:
                sigma = point['sigma']  # later you can handle the keys better
                tau = point['tau']
                mu = point['mu']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        if sigma <= 0:
            return self.NEGINF*torch.ones((*value.shape[:-2], 1, 1),
                                          device=self.device, dtype=self.tch_dtype)  # size (chain_size, 1, 1)

        sampleSize = 1
        if not (value.dim() == 0):  # a scalar tensor has dim = 0
            # value size (chain_size, G, C)
            sampleSize = value.shape[-2]*value.shape[-1]  # value.reshape(-1).shape[0]

        const = torch.log(tau / torch.tensor(np.pi) / 2.)
        logp = (self.torch_sum(-tau * (value - mu)**2,
                               dim=(-2, -1), keepdim=True) + sampleSize * const)/2.  # (chain_size, 1, 1)
        return logp


class Uniform(Distribution):
    def __init__(self, name, lower=0., upper=1., device='cpu', sampler=None, tch_dtype=None):
        super().__init__(name=name, device=device, sampler=sampler, tch_dtype=tch_dtype)
        self.lower = lower
        self.upper = upper
        self.mean = (upper + lower) / 2.
        self.median = self.mean

    def generate_samples(self, size=None, point=None, seed=None):
        lower = self.lower
        upper = self.upper

        if point:
            try:
                lower = point['lower']
                upper = point['upper']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        if size is None:
            print('Uniform is None')
            sampleSize = torch.Size([1])
        elif np.isscalar(size):
            print('Uniform is scalar')
            sampleSize = torch.Size([size])
        else:
            # print(f'Uniform size {size}')
            sampleSize = size

        gen = torch.distributions.uniform.Uniform(low=lower, high=upper)
        samples = self.sampler(gen, sample_shape=sampleSize)
        return samples

    def logp(self, value, point=None):
        lower = self.lower
        upper = self.upper

        if point:
            try:
                lower = point['lower']
                upper = point['upper']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        cond1 = torch.all(torch.all(value >= lower, dim=-1, keepdim=True), dim=-2, keepdim=True)  # (chain_size, 1, 1)
        cond2 = torch.all(torch.all(value <= upper, dim=-1, keepdim=True), dim=-2, keepdim=True)  # (chain_size, 1, 1)
        mask = cond1*cond2

        sampleSize = 1
        if not (value.dim() == 0):
            sampleSize = value.shape[-1]*value.shape[-2]  # value size (chain_size, 1, 1)

        logp = torch.sum(-torch.log(upper - lower))*sampleSize
        logp = torch.ones((*value.shape[:-2], 1, 1),
                          device=self.device, dtype=self.tch_dtype)*logp  # (chain_size, 1, 1)
        logp[torch.logical_not(mask)] = self.NEGINF  # -float('Inf')
        return logp


class HalfCauchy(Distribution):   # check the formula used for sample generation
    def __init__(self, name, beta, device='cpu', sampler=None, torch_sum=None, tch_dtype=None):
        super().__init__(name=name, device=device,
                         sampler=sampler, torch_sum=torch_sum, tch_dtype=tch_dtype)
        self.mode = 0.
        self.median = self.beta = beta  # should be floatX in pymc3 floatX(beta)

    def generate_samples(self, size=None, point=None, seed=None):
        beta = self.beta
        if point:
            try:
                beta = point['beta']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        if size is None:
            print('halfcauchy is None')
            sampleSize = torch.Size([1])
        elif np.isscalar(size):
            print('halfcauchy is scalar')
            sampleSize = torch.Size([size])
        else:
            sampleSize = size

        gen = torch.distributions.half_cauchy.HalfCauchy(scale=beta)
        samples = self.sampler(gen, sample_shape=sampleSize)
        return samples

    def logp(self, value, point=None):
        beta = self.beta
        if point:
            try:
                beta = point['beta']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        cond1 = torch.all(torch.all(value >= 0, dim=-1, keepdim=True), dim=-2, keepdim=True)
        cond2 = beta > 0  # this has only one element see if it is okay
        mask = cond1*cond2

        sampleSize = 1
        if not (value.dim() == 0):
            #sampleSize= value.reshape(-1).shape[0]
            sampleSize = value.shape[-1]*value.shape[-2]  # value size (chain_size, 1, 1)

        const = torch.log(torch.tensor(2.)) - torch.log(torch.tensor(np.pi)) - torch.log(beta)
        logp = self.torch_sum(-torch.log1p((value / beta)**2),
                              dim=(-2, -1), keepdim=True) + sampleSize*const  # (chain_size, 1, 1)
        logp[torch.logical_not(mask)] = self.NEGINF  # -float('Inf')
        return logp


class HalfNormal(Distribution):
    def __init__(self, name, sigma=None, device='cpu', sampler=None, torch_sum=None, tch_dtype=None):
        super().__init__(name=name, device=device,
                         sampler=sampler, torch_sum=torch_sum, tch_dtype=tch_dtype)
        self.sigma = sigma
        self.tau = sigma**-2

        self.mean = torch.sqrt(2. / (torch.tensor(np.pi) * self.tau))
        self.variance = (1. - 2. / torch.tensor(np.pi)) / self.tau

    def generate_samples(self, size=None, point=None, seed=None):
        sigma = self.sigma
        if point:
            try:
                sigma = point['sigma']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        if size is None:
            size = torch.Size([1])
        elif np.isscalar(size):
            size = torch.Size([size])

        gen = torch.distributions.half_normal.HalfNormal(scale=sigma)
        samples = self.sampler(gen, sample_shape=size)
        return samples

    def logp(self, value, point=None):
        tau = self.tau
        sigma = self.sigma

        if point:
            try:
                tau = point['tau']
                sigma = point['sigma']
            except KeyError as e:
                print('KeyError - reason "%s"' % str(e))

        cond1 = torch.all(torch.all(value >= 0, dim=-1, keepdim=True), dim=-2, keepdim=True)
        cond2 = (tau > 0 and sigma > 0)
        mask = cond1*cond2

        sampleSize = 1
        if not (value.dim() == 0):
            # sampleSize= value.reshape(-1).shape[0]
            sampleSize = value.shape[-2]*value.shape[-1]

        const = 0.5 * torch.log(tau * 2. / torch.tensor(np.pi))
        logp = self.torch_sum(-0.5 * tau * value**2,
                              dim=(-2, -1), keepdim=True) + sampleSize * const  # (chain_size, G, C)
        logp[torch.logical_not(mask)] = self.NEGINF  # -float('Inf')
        return logp


# Transformed Distributions:
class TransformedDistribution:
    def __init__(self, name, dist, transform, torch_sum):
        self.name = name
        self.dist = dist
        self.transform = transform
        self.torch_sum = torch_sum

    def logp(self, y, point=None):  # point has the parameters the distribution is conditioned on
        logp_nojac = self.logp_nojac(y, point)  # (chain_size, 1, 1)
        jacobian_det = self.transform.jacobian_det(y)  # it could be (chain_size, 1, 1) or (chain_size, 1, N)
        # print(f'jacobian_det is {jacobian_det}')
        logp = logp_nojac + self.torch_sum(jacobian_det,
                                           dim=[-2, -1], keepdim=True)  # the determinant is returned for each sample
        return logp

    def logp_nojac(self, y, point=None):  # point has the parameters the distribution is conditioned on
        logp_nojac = self.dist.logp(self.transform.backward(y), point)  # (chain_size, 1, 1)
        return logp_nojac

    def generate_samples(self, size=None, point=None, seed=None):
        samples_x = self.dist.generate_samples(size=size, point=point, seed=seed)
        samples_y = self.transform.forward(samples_x)
        return samples_y
