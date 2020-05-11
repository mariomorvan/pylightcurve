import pytest

from pylightcurve.exoplanet_lc_torch import *
from pylightcurve.exoplanet_orbit_torch import *
from pylightcurve.utils import param_sampler, ldc_sampler

torch.set_default_dtype(torch.float64)

METHODS = ['linear', 'quad', 'sqrt', 'claret']

def test_exoplanet_orbit():
    time_array = torch.linspace(0, 10, 20)
    # Numpy compatibility
    from pylightcurve.exoplanet_orbit import exoplanet_orbit as eo_np

    # TENSOR inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=0)
    positions = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True, seed=0)
    positions_np = eo_np(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array.numpy())
    for i in range(len(positions)):
        assert np.allclose(positions[i].numpy(), positions_np[i])


def test_transit_projected_distance():
    time_array = torch.linspace(0, 10, 20)

    # Scalar inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True, seed=7)

    # Numpy compatibility
    from pylightcurve.exoplanet_orbit import transit_projected_distance as tpd_np
    distances_np = tpd_np(period, sma_over_rs, eccentricity,
                          inclination, periastron, mid_time, time_array.numpy())

    # Tensor inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=7)
    distances = transit_projected_distance(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                                           time_array)
    assert isinstance(distances, torch.Tensor)
    assert np.allclose(distances.numpy(), distances_np, atol=1.e-7)


def test_transit_duration():
    # Tensor inputs
    rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler()
    duration = transit_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron)
    assert isinstance(duration, torch.Tensor)

    # Numpy compat
    from pylightcurve.exoplanet_orbit import transit_duration as td_np
    duration_np = td_np(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron)
    assert np.allclose(duration.numpy(), duration_np)


def test_integral_r():
    r = torch.rand(10)
    seed = np.random.randint(1e9)
    from pylightcurve.exoplanet_lc import integral_r as integral_r_np
    for method in integral_r:
        result = integral_r[method](ldc_sampler(method, seed=seed), r)
        result_np = integral_r_np[method](ldc_sampler(method, out_list=True, seed=seed), r.numpy())
        assert np.allclose(result, result_np)


def test_integral_r_f():
    r1 = torch.rand(10)
    r2 = torch.rand(10)
    rprs = torch.rand(10)
    z = torch.rand(10)
    seed = np.random.randint(1e9)
    from pylightcurve.exoplanet_lc import integral_r_f as integral_r_f_np
    for method in integral_r_f:
        result = integral_r_f[method](ldc_sampler(method, seed=seed), rprs, z, r1, r2)
        result_np = integral_r_f_np[method](ldc_sampler(method, out_list=True, seed=seed), rprs.numpy(), z.numpy(), r1.numpy(), r2.numpy())
        assert np.allclose(np.nanmean(result.numpy(), 0), np.nanmean(result_np, 0))


def test_integral_minus_core():
    rp_over_rs = torch.rand(1)
    z = torch.rand(10)
    ww1 = torch.rand(10)
    ww2 = torch.rand(10)
    seed = np.random.randint(1e9)
    from pylightcurve.exoplanet_lc import integral_minus_core as imc_np
    for method in integral_r:
        result = integral_minus_core(method, ldc_sampler(method, seed=seed), rp_over_rs, z, ww1, ww2)
        result_np = imc_np(method, ldc_sampler(method, out_list=True, seed=seed), rp_over_rs.numpy(), z.numpy(), ww1.numpy(), ww2.numpy())
        assert np.allclose(np.nanmean(result.numpy(), 0), np.nanmean(result_np, 0))


def test_integral_plus_core():
    rp_over_rs = torch.rand(1)
    z = torch.rand(10)
    ww1 = torch.rand(10)
    ww2 = torch.rand(10)
    seed = np.random.randint(1e9)
    from pylightcurve.exoplanet_lc import integral_plus_core as ipc_np
    for method in integral_r:
        result = integral_plus_core(method, ldc_sampler(method, seed=seed), rp_over_rs, z, ww1, ww2)
        result_np = ipc_np(method, ldc_sampler(method, out_list=True, seed=seed), rp_over_rs.numpy(), z.numpy(), ww1.numpy(), ww2.numpy())
        assert np.allclose(np.nanmean(result.numpy(), 0), np.nanmean(result_np, 0))


def test_transit_flux_drop():
    rp_over_rs = torch.rand(1)
    z_over_rs = torch.linspace(0, 1, 10)
    seed = np.random.randint(1e9)

    from pylightcurve.exoplanet_lc import transit_flux_drop as transit_flux_drop_np
    for method in integral_r:
        result = transit_flux_drop(method, ldc_sampler(method, seed=seed), rp_over_rs, z_over_rs)
        result_np = transit_flux_drop_np(method, ldc_sampler(method, out_list=True, seed=seed), rp_over_rs.numpy(), z_over_rs)

        assert not torch.isnan(result).all()
        assert np.allclose(np.nanmean(result.numpy()[:, None], 1), np.nanmean(result_np[:, None], 1))


def test_transit():
    from pylightcurve.exoplanet_lc import transit as transit_np
    time_array = torch.linspace(0, 10, 20)

    seed = np.random.randint(1e9)

    for method in METHODS:
        rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=1)
        result = transit(method, ldc_sampler(method, seed=seed), rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
                         mid_time, time_array, precision=3)

        rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(
            out_scalar=True, seed=1)
        result_np = transit_np(method, ldc_sampler(method, out_list=True, seed=seed), rp_over_rs, period, sma_over_rs, eccentricity, inclination,
                               periastron,
                               mid_time, time_array.numpy(), precision=3)

        assert np.allclose(result.numpy(), result_np)


def test_transit_perf():
    import time
    from pylightcurve.exoplanet_lc import transit as transit_np
    time_array = torch.linspace(0, 10, 100)
    seed = np.random.randint(1e9)
    for method in METHODS:
        rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=seed)

        t0 = time.time()
        for i in range(100):
            transit(method, ldc_sampler(method, seed=seed), rp_over_rs, period, sma_over_rs, eccentricity, inclination,
                    periastron,
                    mid_time, time_array, precision=3)
        dur_torch = time.time() - t0

        rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(
            out_scalar=True, seed=seed)
        t0 = time.time()
        for i in range(100):
            transit_np(method, ldc_sampler(method, out_list=True, seed=seed), rp_over_rs, period, sma_over_rs, eccentricity, inclination,
                       periastron,
                       mid_time, time_array.numpy(), precision=3)
        dur_np = time.time() - t0

        # Simply sanity check of factor 10... gotta investigate though
        assert dur_torch < 10 * dur_np
        print('pytorch perf', dur_torch)
        print('numpy perf', dur_np)


def test_eclipse():
    from pylightcurve.exoplanet_lc import eclipse as eclipse_np
    time_array = torch.linspace(0, 10, 20)

    rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=1)
    result = eclipse(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
                     mid_time, time_array, precision=3)

    rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(
        out_scalar=True, seed=1)
    result_np = eclipse_np(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
                           mid_time, time_array.numpy(), precision=3)

    assert np.allclose(result.numpy(), result_np)


def test_transit_grad():
    torch.autograd.detect_anomaly()

    time_array = torch.linspace(0, 10, 20)

    for method in METHODS:

        # Testing backward gradient autograd compatibility
        param_dict = param_sampler(seed=1, requires_grad=True, return_dict=True)
        param_dict.pop('fp_over_fs')
        flux = transit(method, ldc_sampler(method, requires_grad=True), time_array=time_array, **param_dict, precision=3)

        result = (flux ** 2).sum()
        result.backward()
        for p, v in param_dict.items():
            grad_p = v.grad
            assert grad_p is not None and not torch.isnan(grad_p).item()


def test_eclipse_grad():
    torch.autograd.detect_anomaly()

    time_array = torch.linspace(0, 10, 20)

    # Testing backward gradient autograd compatibility
    param_dict = param_sampler(seed=1, requires_grad=True, return_dict=True)
    flux = eclipse(time_array=time_array, **param_dict, precision=3)

    result = (flux ** 2).sum()
    result.backward()
    for p, v in param_dict.items():
        grad_p = v.grad
        assert grad_p is not None and not torch.isnan(grad_p).item()



def test_gpu():
    #torch.cuda.manual_seed_all()

    if not torch.cuda.is_available():
        pytest.skip('no gpu available')

    time_array = torch.linspace(0, 10, 20).cuda()
    param_dict = param_sampler(requires_grad=True, return_dict=True, device='cuda')
    ldc = ldc_sampler('linear', device='cuda')

    # Transit
    param_dict.pop('fp_over_fs')
    flux = transit('linear', ldc, time_array=time_array, **param_dict, precision=3)
