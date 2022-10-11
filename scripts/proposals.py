import torch


# Proposals:
# proposal distributions:
class Proposal:
    def __init__(self, s, sampler=None):
        self.s = s
        self.sampler = sampler


# tensor versions:
class NormalProposal(Proposal):
    def __call__(self):
        # self.s should be a tensor and the size of the sample
        if self.sampler.is_batch:
            norm_samps = self.sampler.normal(sample_shape=self.s.shape)
            norm_samps *= self.s
            return norm_samps
        else:
            return torch.normal(mean=torch.tensor(0., device=self.s.device, dtype=self.s.dtype), std=self.s)


class MultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = torch.linalg.cholesky(s, lower=True)

    def __call__(self, num_draws=None):
        assert False, 'use_batch_rng = True is not implemented. do not use this'
        if num_draws is not None:
            b = torch.randn(self.n, num_draws)
            return torch.transpose(torch.dot(self.chol, b), dim0=0, dim1=1)
        else:
            b = torch.randn(self.n)
            return torch.dot(self.chol, b)
