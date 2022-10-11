import torch
from mymath import invlogit, softplus


class Transform:
    def forward(self, x):
        return

    def backward(self, y):
        return

    def jacobian_det(self, y_):
        return


class Log:
    name = "log"

    def backward(self, y):
        return torch.exp(y)

    def forward(self, x):
        if torch.any(x < 0):
            print(f"in log reached neegative")
        return torch.log(x)

    def jacobian_det(self, y):
        return y  # the log of the det is returned


class StickBreaking(Transform):
    # make sure the input to the functions has float32 or 64 as dtype
    name = "stickbreaking"

    def __init__(self, eps=1e-9, device='cpu', torch_sum=None):
        self.eps = eps
        self.device = device
        self.torch_sum = torch_sum

    def forward(self, x):
        # should work with size(x)= C*N, N is the number of samples
        if not (x.dtype == torch.float32 or x.dtype == torch.float64):  # there is no torch.float128
            raise ValueError("change dtype to float32 or float64")
        mb_size = x.shape[:-2]  # which is chain_size in case only chain is the added dimension
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        # reverse cumsum
        x0 = x[..., :-1, :]  # excluding the last row, size (chain_size, C-1, N)
        x0_flipped = torch.flip(x0, dims=[-2])  # flip along the rows which is -2 in batch added W (chain_size, C, N)

        x_lastrow_ext = x[..., -1, :]  # size (chain_size, N)
        x_lastrow_ext = x_lastrow_ext.unsqueeze(dim=-2)  # size (chain_size, 1, N)
        s = torch.flip(torch.cumsum(x0_flipped, dim=-2), dims=[-2]) + x_lastrow_ext  # size (chain_size, C-1, N)
        z = x[..., :-1, :]/s  # (chain_size, C-1, N)

        # to stabilize logit where z = 1:
        z[z == 1] = 1-1e-15
        K, samples = x.shape[-2:]  # last two dimension C, N
        Kvec = torch.arange(1., K, 1., dtype=x.dtype, device=self.device).reshape(-1, 1)
        Kvec = Kvec.repeat((*mb_size, 1, samples))
        y = torch.logit(z) - torch.logit(1.0 / (K - Kvec + 1.))
        return y

    def backward(self, y):
        if not (y.dtype == torch.float32 or y.dtype == torch.float64):
            raise ValueError("change dtype to float32 or float64")

        if y.dim() == 1:
            y = y.reshape((-1, 1))

        mb_size = y.shape[:-2]
        ydim, samples = y.shape[-2:]  # (chain_size, C-1, N)
        K = ydim + 1.
        Kvec = torch.arange(1., K, 1., dtype=y.dtype, device=self.device).reshape(-1, 1)
        Kvec = Kvec.repeat((*mb_size, 1, samples))
        y_trans = y + torch.logit(1.0 / (K - Kvec + 1.))

        z = invlogit(y_trans, eps=self.eps)  # size(z)= (K-1, 1) # (chain_size, C-1, N)
        z_inv = 1.-z

        ones_arr = torch.ones((*mb_size, 1, samples), dtype=y.dtype, device=self.device)
        z_ = torch.cat((ones_arr, z_inv), dim=-2)  # (chain_size, C, N)
        z_prod = torch.cumprod(z_, dim=-2)
        z_ext = torch.cat((z, ones_arr), dim=-2)
        x = z_ext * z_prod
        return x

    def jacobian_det(self, y):
        if not (y.dtype == torch.float32 or y.dtype == torch.float64):
            raise ValueError("change dtype to float32 or float64")

        if y.dim() == 1:
            y = y.reshape((-1, 1))
        mb_size = y.shape[:-2]
        ydim, samples = y.shape[-2:]

        K = ydim + 1.
        Kvec = torch.arange(1., K, 1., dtype=y.dtype, device=self.device).reshape(-1, 1)
        Kvec = Kvec.repeat((*mb_size, 1, samples))
        y_trans = y + torch.logit(1.0 / (K - Kvec + 1.))
        # then exp(y_trans) could become inf, and you should catch the warning

        z = invlogit(y_trans, eps=self.eps)  # size(z)= (K-1, 1) # (chain_size, C-1, N)
        z_inv = 1.-z

        ones_arr = torch.ones((*mb_size, 1, samples), dtype=y.dtype, device=self.device)
        z_ = torch.cat((ones_arr, z_inv), dim=-2)  # (chain_size, C, N)

        z_prod = torch.cumprod(z_, dtype=y.dtype, dim=-2)  # (chain_size, C, N)
        det = self.torch_sum(torch.log(z_prod[..., :-1, :]) - torch.log1p(torch.exp(y_trans)) - torch.log1p(torch.exp(-y_trans)),
                             dim=-2, keepdim=True)  # (chain_size, 1, N)
        assert det.shape[-1] == samples, "mismatch in size"
        assert det.shape[:-2] == mb_size, "mismatch in size"

        return det


class Interval(Transform):
    """Transform from real line interval [a,b] to whole real line."""

    name = "interval"

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, y):
        a, b = self.a, self.b
        x = (b - a) * torch.sigmoid(y) + a
        return x

    def forward(self, x):
        a, b = self.a, self.b
        y = torch.log(x - a) - torch.log(b - x)
        return y

    def jacobian_det(self, y):
        soft_ = softplus(-y)
        det = torch.log(self.b - self.a) - y - 2*soft_
        return det
