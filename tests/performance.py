from timeit import default_timer as timer

import numpy  as np
import torch

from pylightcurve.utils import param_sampler, ldc_sampler

torch.set_default_dtype(torch.float64)


def measure_perf_transit_numpy(array_length=500, repeat=10, method='claret', precision=3, seed=None):
    # from pylightcurve.exoplanet_orbit import *
    from pylightcurve.exoplanet_lc import exoplanet_orbit, transit_flux_drop

    (rp_over_rs, fp_over_fs, period, sma_over_rs,
     eccentricity, inclination, periastron, mid_time) = param_sampler(out_scalar=True, seed=seed)
    limb_darkening_coefficients = ldc_sampler(method=method, seed=seed, out_list=True)
    time_array = np.linspace(0, 20, int(array_length))

    t0 = timer()
    for _ in range(repeat):
        position_vector = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                                          time_array)

    t1 = timer()

    for _ in range(repeat):
        projected_distance = np.where(
            position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
            np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    t2 = timer()

    for _ in range(repeat):
        transit_flux_drop(method, limb_darkening_coefficients, rp_over_rs, projected_distance, precision=precision)

    t3 = timer()

    return np.diff([t0, t1, t2, t3])


def measure_perf_transit_torch(array_length=500, repeat=10, method='claret', precision=3, seed=None):
    # from pylightcurve.exoplanet_orbit_torch import *
    from pylightcurve.exoplanet_lc_torch import exoplanet_orbit, transit_flux_drop

    rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(
        seed=seed)
    limb_darkening_coefficients = ldc_sampler(method=method, seed=seed)

    time_array = torch.linspace(0, 20., int(array_length))

    t0 = timer()

    for _ in range(repeat):
        position_vector = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                                          time_array)

    t1 = timer()

    for _ in range(repeat):
        projected_distance = torch.where(
            position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
            torch.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    t2 = timer()
    for _ in range(repeat):
        transit_flux_drop(method, limb_darkening_coefficients, rp_over_rs, projected_distance, precision=precision)

    t3 = timer()

    return np.diff([t0, t1, t2, t3])


def measure_perf_flux_drop_torch(array_length=500, repeat=10, method='claret', precision=3, seed=None):
    from pylightcurve.exoplanet_lc_torch import exoplanet_orbit, transit_flux_drop, integral_plus_core, integral_centred, integral_minus_core

    rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(
        seed=seed)
    limb_darkening_coefficients = ldc_sampler(method=method, seed=seed)

    time_array = torch.linspace(0, 20., int(array_length))



    for _ in range(repeat):
        position_vector = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                                          time_array)

    for _ in range(repeat):
        projected_distance = torch.where(
            position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
            torch.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))
    z_over_rs = projected_distance


    times = []
    times += [timer()]
    time_labels = []

    # Start

    if len(z_over_rs) == 0:
        return torch.Tensor([])

    z_over_rs = torch.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)

    times += [timer()]
    time_labels += ['z_over_rs initiliazation']


    # cases
    zsq = z_over_rs * z_over_rs
    sum_z_rprs = z_over_rs + rp_over_rs
    dif_z_rprs = rp_over_rs - z_over_rs
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2

    times += [timer()]
    time_labels +=['cases prepar']

    case0 = torch.where((z_over_rs == 0) & (rp_over_rs <= 1))
    case1 = torch.where((z_over_rs < rp_over_rs) & (sum_z_rprs <= 1))
    casea = torch.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1))
    caseb = torch.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1))
    case2 = torch.where((z_over_rs == rp_over_rs) & (sum_z_rprs <= 1))
    casec = torch.where((z_over_rs == rp_over_rs) & (sum_z_rprs > 1))
    case3 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs < 1))
    case4 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs == 1))
    case5 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1))
    case6 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1))
    case7 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs))

    times += [timer()]
    time_labels +=['cases comput']

    plus_case = torch.cat((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
    minus_case = torch.cat((case3[0], case4[0], case5[0], case6[0], case7[0]))
    star_case = torch.cat((case5[0], case6[0], case7[0], casea[0], casec[0]))

    times += [timer()]
    time_labels +=['case concat']

    # cross points
    ph = torch.acos(torch.clamp((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), -1, 1))
    theta_1 = torch.zeros(len(z_over_rs))
    ph_case = torch.cat((case5[0], casea[0], casec[0]))

    times += [timer()]
    time_labels +=['cross points 1']

    theta_1[ph_case] = ph[ph_case]
    theta_2 = torch.asin(torch.min(rp_over_rs / z_over_rs, torch.ones_like(z_over_rs)))
    theta_2[case1] = np.pi
    theta_2[case2] = np.pi / 2.0
    theta_2[casea] = np.pi
    theta_2[casec] = np.pi / 2.0
    theta_2[case7] = ph[case7]

    times += [timer()]
    time_labels +=['cross points 2']

    # flux_upper
    plusflux = torch.zeros(len(z_over_rs))

    times += [timer()]
    time_labels +=['flux upper 1']

    plusflux[plus_case] = integral_plus_core(method, limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
                                             theta_1[plus_case], theta_2[plus_case], precision=precision)
    times += [timer()]
    time_labels +=['flux upper 2']

    if len(case0[0]) > 0:
         plusflux[case0] = integral_centred(method, limb_darkening_coefficients, rp_over_rs, torch.zeros(1), np.pi)
    if len(caseb[0]) > 0:
         plusflux[caseb] = integral_centred(method, limb_darkening_coefficients, torch.ones(1), torch.zeros(1), np.pi)
    times += [timer()]
    time_labels +=['flux upper 3']

    # flux_lower
    minsflux = torch.zeros(len(z_over_rs))
    minsflux[minus_case] = integral_minus_core(method, limb_darkening_coefficients, rp_over_rs,
                                               z_over_rs[minus_case], torch.zeros(1), theta_2[minus_case], precision=precision)
    times += [timer()]
    time_labels +=['flux lower']

    # flux_star
    starflux = torch.zeros(len(z_over_rs))
    starflux[star_case] = integral_centred(method, limb_darkening_coefficients, torch.ones(1), torch.zeros(1), ph[star_case])
    times += [timer()]
    time_labels += ['flux star']

    # flux_total
    total_flux = integral_centred(method, limb_darkening_coefficients, torch.ones(1), torch.zeros(1), 2.0 * np.pi)

    times += [timer()]
    time_labels += ['total flux']

    return np.diff(times), time_labels


def measure_perf_flux_drop_numpy(array_length=500, repeat=10, method='claret', precision=3, seed=None):
    from pylightcurve.exoplanet_lc import exoplanet_orbit, integral_plus_core, integral_centred, integral_minus_core

    rp_over_rs, fp_over_fs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time = param_sampler(
        out_scalar=True, seed=seed)
    limb_darkening_coefficients = ldc_sampler(method=method, seed=seed, out_list=True)

    time_array = np.linspace(0, 20., int(array_length))



    for _ in range(repeat):
        position_vector = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                                          time_array)

    for _ in range(repeat):
        projected_distance = np.where(
            position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
            np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))
    z_over_rs = projected_distance


    times = []
    times += [timer()]
    time_labels = []

    # Start

    if len(z_over_rs) == 0:
        return torch.Tensor([])

    z_over_rs = np.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)

    times += [timer()]
    time_labels += ['z_over_rs initiliazation']


    # cases
    zsq = z_over_rs * z_over_rs
    sum_z_rprs = z_over_rs + rp_over_rs
    dif_z_rprs = rp_over_rs - z_over_rs
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2

    times += [timer()]
    time_labels +=['cases prepar']

    case0 = torch.where((z_over_rs == 0) & (rp_over_rs <= 1))
    case1 = torch.where((z_over_rs < rp_over_rs) & (sum_z_rprs <= 1))
    casea = torch.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1))
    caseb = torch.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1))
    case2 = torch.where((z_over_rs == rp_over_rs) & (sum_z_rprs <= 1))
    casec = torch.where((z_over_rs == rp_over_rs) & (sum_z_rprs > 1))
    case3 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs < 1))
    case4 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs == 1))
    case5 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1))
    case6 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1))
    case7 = torch.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs))

    times += [timer()]
    time_labels +=['cases comput']

    plus_case = torch.cat((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
    minus_case = torch.cat((case3[0], case4[0], case5[0], case6[0], case7[0]))
    star_case = torch.cat((case5[0], case6[0], case7[0], casea[0], casec[0]))

    times += [timer()]
    time_labels +=['case concat']

    # cross points
    ph = torch.acos(torch.clamp((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), -1, 1))
    theta_1 = torch.zeros(len(z_over_rs))
    ph_case = torch.cat((case5[0], casea[0], casec[0]))

    times += [timer()]
    time_labels +=['cross points 1']

    theta_1[ph_case] = ph[ph_case]
    theta_2 = torch.asin(torch.min(rp_over_rs / z_over_rs, torch.ones_like(z_over_rs)))
    theta_2[case1] = np.pi
    theta_2[case2] = np.pi / 2.0
    theta_2[casea] = np.pi
    theta_2[casec] = np.pi / 2.0
    theta_2[case7] = ph[case7]

    times += [timer()]
    time_labels +=['cross points 2']

    # flux_upper
    plusflux = torch.zeros(len(z_over_rs))

    times += [timer()]
    time_labels +=['flux upper 1']

    plusflux[plus_case] = integral_plus_core(method, limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
                                             theta_1[plus_case], theta_2[plus_case], precision=precision)
    times += [timer()]
    time_labels +=['flux upper 2']

    if len(case0[0]) > 0:
         plusflux[case0] = integral_centred(method, limb_darkening_coefficients, rp_over_rs, torch.zeros(1), np.pi)
    if len(caseb[0]) > 0:
         plusflux[caseb] = integral_centred(method, limb_darkening_coefficients, torch.ones(1), torch.zeros(1), np.pi)
    times += [timer()]
    time_labels +=['flux upper 3']

    # flux_lower
    minsflux = torch.zeros(len(z_over_rs))
    minsflux[minus_case] = integral_minus_core(method, limb_darkening_coefficients, rp_over_rs,
                                               z_over_rs[minus_case], torch.zeros(1), theta_2[minus_case], precision=precision)
    times += [timer()]
    time_labels +=['flux lower']

    # flux_star
    starflux = torch.zeros(len(z_over_rs))
    starflux[star_case] = integral_centred(method, limb_darkening_coefficients, torch.ones(1), torch.zeros(1), ph[star_case])
    times += [timer()]
    time_labels += ['flux star']

    # flux_total
    total_flux = integral_centred(method, limb_darkening_coefficients, torch.ones(1), torch.zeros(1), 2.0 * np.pi)

    times += [timer()]
    time_labels += ['total flux']

    return np.diff(times), time_labels

