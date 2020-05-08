from pylightcurve.utils import param_sampler, ldc_sampler
from pylightcurve.exoplanet_lc_torch import *
from pylightcurve.exoplanet_orbit_torch import *
torch.set_default_dtype(torch.float64)

ldc_torch = {
    'claret': torch.rand(4),
    'linear': torch.rand(1),
    'quad': torch.rand(2),
    'sqrt': torch.rand(2)
}

ldc = {
    'claret': ldc_torch['claret'].numpy().tolist(),
    'linear': ldc_torch['linear'].numpy().tolist(),
    'quad': ldc_torch['quad'].numpy().tolist(),
    'sqrt': ldc_torch['sqrt'].numpy().tolist()
}




def test_exoplanet_orbit():
    time_array = torch.linspace(0, 10, 20)

    # Scalar inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True)
    #period, sma_over_rs, eccentricity, inclination, periastron, mid_time = 1, 2, 0, 0, 0, 0
    positions = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)


    # Numpy compatibility
    from pylightcurve.exoplanet_orbit import exoplanet_orbit as eo_np
    positions_np = eo_np(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array.numpy())
    for i in range(len(positions)):
        assert np.allclose(positions[i].numpy(), positions_np[i])

    # TENSOR inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time  = param_sampler(seed=0)
    positions = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True, seed=0)
    positions_np = eo_np(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array.numpy())
    for i in range(len(positions)):
        assert np.allclose(positions[i].numpy(), positions_np[i])

def test_transit_projected_distance():
    time_array = torch.linspace(0, 10, 20)

    # Scalar inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True)
    distances = transit_projected_distance(period, sma_over_rs, eccentricity,
                                           inclination, periastron, mid_time, time_array)
    assert isinstance(distances, torch.Tensor)

    # Numpy compatibility
    from pylightcurve.exoplanet_orbit import transit_projected_distance as tpd_np
    distances_np = tpd_np(period, sma_over_rs, eccentricity,
                          inclination, periastron, mid_time, time_array.numpy())
    assert np.allclose(distances.numpy(), distances_np, atol=1.e-7)

    # Tensor inputs
    _, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler()
    distances = transit_projected_distance(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                                           time_array)
    assert isinstance(distances, torch.Tensor)


def test_transit_duration():
    # Scalar value
    rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True)
    duration = transit_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron)
    assert isinstance(duration, float)

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

    from pylightcurve.exoplanet_lc import integral_r as integral_r_np
    for method in integral_r:
        result = integral_r[method](ldc[method], r)
        result_np = integral_r_np[method](ldc[method], r.numpy())
        assert np.allclose(result, result_np)


def test_integral_r_f():
    r1 = torch.rand(10)
    r2 = torch.rand(10)
    rprs = torch.rand(10)
    z = torch.rand(10)

    from pylightcurve.exoplanet_lc import integral_r_f as integral_r_f_np
    for method in integral_r_f:
        result = integral_r_f[method](ldc[method], rprs, z, r1, r2)
        result_np = integral_r_f_np[method](ldc[method], rprs.numpy(), z.numpy(), r1.numpy(), r2.numpy())
        assert np.allclose( np.nanmean(result.numpy(), 0), np.nanmean(result_np, 0))



def test_integral_minus_core():
    rp_over_rs = torch.rand(1)
    z = torch.rand(10)
    ww1 = torch.rand(10)
    ww2 = torch.rand(10)

    from pylightcurve.exoplanet_lc import integral_minus_core as imc_np
    for method in integral_r:
        result = integral_minus_core(method, ldc_torch[method], rp_over_rs, z, ww1, ww2)
        result_np = imc_np(method, ldc[method], rp_over_rs.numpy(), z.numpy(), ww1.numpy(), ww2.numpy())
        assert np.allclose(np.nanmean(result.numpy(), 0), np.nanmean(result_np,0))

def test_integral_plus_core():
    rp_over_rs = torch.rand(1)
    z = torch.rand(10)
    ww1 = torch.rand(10)
    ww2 = torch.rand(10)

    from pylightcurve.exoplanet_lc import integral_plus_core as ipc_np
    for method in integral_r:
        result = integral_plus_core(method, ldc_torch[method], rp_over_rs, z, ww1, ww2)
        result_np = ipc_np(method, ldc[method], rp_over_rs.numpy(), z.numpy(), ww1.numpy(), ww2.numpy())
        assert np.allclose(np.nanmean(result.numpy(), 0), np.nanmean(result_np, 0))

def test_transit_flux_drop():
    rp_over_rs = torch.rand(1)
    z_over_rs = torch.linspace(0, 1, 10)

    from pylightcurve.exoplanet_lc import transit_flux_drop as transit_flux_drop_np
    for method in integral_r:
        result = transit_flux_drop(method, ldc_torch[method], rp_over_rs, z_over_rs)
        result_np = transit_flux_drop_np(method, ldc[method], rp_over_rs.numpy(), z_over_rs)

        assert not torch.isnan(result).all()
        assert np.allclose(np.nanmean(result.numpy()[:,None], 1), np.nanmean(result_np[:,None], 1))

def test_transit():
    from pylightcurve.exoplanet_lc import transit as transit_np
    time_array = torch.linspace(0, 10, 20)

    for method in ldc_torch:

        rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=1)
        result = transit(method, ldc[method], rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
                         mid_time, time_array, precision=3)

        rp_over_rs, _, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True, seed=1)
        result_np = transit_np(method, ldc[method], rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
                  mid_time, time_array.numpy(), precision=3)

        assert np.allclose(result.numpy(), result_np)


def test_transit_perf():
    import time
    from pylightcurve.exoplanet_lc import transit as transit_np
    time_array = torch.linspace(0, 10, 100)

    for method in ldc_torch:
        seed = 0 #np.random.randint(0)
        rp_over_rs, _ , period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(seed=seed)

        t0 = time.time()
        for i in range(100):
            transit(method, ldc[method], rp_over_rs, period, sma_over_rs, eccentricity, inclination,
                    periastron,
                    mid_time, time_array, precision=3)
        dur_torch = time.time() - t0

        rp_over_rs, _ ,period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True, seed=seed)
        t0 = time.time()
        for i in range(100):
            transit_np(method, ldc[method], rp_over_rs, period, sma_over_rs, eccentricity, inclination,
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

    rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(out_scalar=True, seed=1)
    result_np = eclipse_np(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
              mid_time, time_array.numpy(), precision=3)

    assert np.allclose(result.numpy(), result_np)