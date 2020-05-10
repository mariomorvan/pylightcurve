import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def param_sampler(*size, out_numpy=False, out_scalar=False, return_dict=False,
                  seed=None, dtype=torch.get_default_dtype(), requires_grad=False):
    """
    returned params:  rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time
    tuple by default, dict if return_dict is set to True
    """
    names = 'rp_over_rs', 'fp_over_fs', 'period', 'sma_over_rs', 'eccentricity', 'inclination', 'periastron', 'mid_time'
    assert not (out_numpy and out_scalar)
    if seed is None:
        seed = np.random.randint(10e6)
    if not size:
        size = 1
    torch.manual_seed(seed)
    rp_over_rs = torch.rand(size, dtype=dtype) / 10
    fp_over_fs = torch.rand(size, dtype=dtype) / 10
    period = torch.rand(size, dtype=dtype) * 10
    sma_over_rs = torch.rand(size, dtype=dtype) * 10 + 1
    eccentricity = torch.rand(size, dtype=dtype)
    inclination = 90 + (torch.rand(size, dtype=dtype) - 0.5) * 4
    periastron = torch.rand(size, dtype=dtype) * 360
    mid_time = torch.rand(size, dtype=dtype) * period
    params = rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time
    if requires_grad:
        for p in params:
            p.requires_grad = True
        params[-1].retain_grad()
    if out_numpy:
        params = tuple([p.numpy() for p in params])
    elif out_scalar:
        params = tuple([p.item() for p in params])
    if return_dict:
        out_dict = dict()
        for i, name in enumerate(names):
            out_dict[name] = params[i]
        return out_dict
    return params


def ldc_sampler(method='claret', out_list=False, seed=None, requires_grad=False):
    if seed is None:
        seed = np.random.randint(10e6)
    torch.manual_seed(seed)
    if method == "claret":
        out = torch.rand(4, requires_grad=requires_grad)
    elif method == "linear":
        out = torch.rand(1, requires_grad=requires_grad)
    elif method == "quad" or method == 'sqrt':
        out = torch.rand(2, requires_grad=requires_grad)
    if out_list:
        out = out.numpy().tolist()
    return out
