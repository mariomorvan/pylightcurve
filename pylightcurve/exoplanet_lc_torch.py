from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .analysis_gauss_numerical_integration_torch import *
from .exoplanet_orbit_torch import *

torch.set_default_dtype(torch.float64)

EPS = 1.e-16
#TODO: Investigate the impact of EPS


def integral_r_claret(limb_darkening_coefficients, r):
    a1, a2, a3, a4 = limb_darkening_coefficients
    mu44 = 1.0 - r * r
    mu24 = torch.sqrt(mu44)
    mu14 = torch.sqrt(mu24)
    return - (2.0 * (1.0 - a1 - a2 - a3 - a4) / 4) * mu44 \
           - (2.0 * a1 / 5) * mu44 * mu14 \
           - (2.0 * a2 / 6) * mu44 * mu24 \
           - (2.0 * a3 / 7) * mu44 * mu24 * mu14 \
           - (2.0 * a4 / 8) * mu44 * mu44


def num_claret(r, limb_darkening_coefficients, rprs, z):
    a1, a2, a3, a4 = limb_darkening_coefficients
    rsq = r * r
    mu44 = 1.0 - rsq
    mu24 = torch.sqrt(mu44)
    mu14 = torch.sqrt(mu24)
    return ((1.0 - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), max=1.0 - EPS))


def integral_r_f_claret(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_claret, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for linear method


def integral_r_linear(limb_darkening_coefficients, r):
    a1 = limb_darkening_coefficients[0]
    musq = 1 - r * r
    return (-1.0 / 6) * musq * (3.0 + a1 * (-3.0 + 2.0 * torch.sqrt(musq)))


def num_linear(r, limb_darkening_coefficients, rprs, z):
    a1 = limb_darkening_coefficients[0]
    rsq = r * r
    return (1.0 - a1 * (1.0 - torch.sqrt(1.0 - rsq))) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), max=1.0 - EPS))


def integral_r_f_linear(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_linear, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for quadratic method


def integral_r_quad(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients[:2]
    musq = 1 - r * r
    mu = torch.sqrt(musq)
    return (1.0 / 12) * (-4.0 * (a1 + 2.0 * a2) * mu * musq + 6.0 * (-1 + a1 + a2) * musq + 3.0 * a2 * musq * musq)


def num_quad(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients[:2]
    rsq = r * r
    cc = 1.0 - torch.sqrt(1.0 - rsq)
    return (1.0 - a1 * cc - a2 * cc * cc) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), max=1.0 - EPS))


def integral_r_f_quad(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_quad, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for square root method


def integral_r_sqrt(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients[:2]
    musq = 1 - r * r
    mu = torch.sqrt(musq)
    return ((-2.0 / 5) * a2 * torch.sqrt(mu) - (1.0 / 3) * a1 * mu + (1.0 / 2) * (-1 + a1 + a2)) * musq


def num_sqrt_torch(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients[:2]
    rsq = r * r
    mu = torch.sqrt(1.0 - rsq)
    return (1.0 - a1 * (1 - mu) - a2 * (1.0 - torch.sqrt(mu))) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), max=1.0 - EPS))


def integral_r_f_sqrt_torch(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_sqrt_torch, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# dictionaries containing the different methods,
# if you define a new method, include the functions in the dictionary as well

integral_r = {
    'claret': integral_r_claret,
    'linear': integral_r_linear,
    'quad': integral_r_quad,
    'sqrt': integral_r_sqrt
}

integral_r_f = {
    'claret': integral_r_f_claret,
    'linear': integral_r_f_linear,
    'quad': integral_r_f_quad,
    'sqrt': integral_r_f_sqrt_torch,
}

num = {
    'claret': num_claret,
    'linear': num_linear,
    'quad': num_quad,
    'sqrt': num_sqrt_torch
}


def integral_centred(method, limb_darkening_coefficients, rprs, ww1, ww2):
    return (integral_r[method](limb_darkening_coefficients, rprs)
            - integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1))) * torch.abs(ww2 - ww1)


def integral_plus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z
    rr1 = z * torch.cos(ww1) + torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww1)) ** 2, EPS))
    rr1 = torch.clamp(rr1, EPS, 1 - EPS)
    rr2 = z * torch.cos(ww2) + torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww2)) ** 2, EPS))
    rr2 = torch.clamp(rr2, EPS, 1 - EPS)
    w1 = torch.min(ww1, ww2)
    r1 = torch.min(rr1, rr2)
    w2 = torch.max(ww1, ww2)
    r2 = torch.max(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1)) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * w2
    partc = integral_r[method](limb_darkening_coefficients, r2) * (-w1)
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc + partd


def integral_minus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z
    rr1 = z * torch.cos(ww1) - torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww1)) ** 2, EPS))
    rr1 = torch.clamp(rr1, EPS, 1 - EPS)
    rr2 = z * torch.cos(ww2) - torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww2)) ** 2, EPS))
    rr2 = torch.clamp(rr2, EPS, 1 - EPS)
    w1 = torch.min(ww1, ww2)
    r1 = torch.min(rr1, rr2)
    w2 = torch.max(ww1, ww2)
    r2 = torch.max(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1)) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * (-w1)
    partc = integral_r[method](limb_darkening_coefficients, r2) * w2
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc - partd


def transit_flux_drop(method, limb_darkening_coefficients, rp_over_rs, z_over_rs, precision=3):
    if len(z_over_rs) == 0:
        return torch.Tensor([])

    z_over_rs = torch.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)

    # cases
    zsq = z_over_rs * z_over_rs
    sum_z_rprs = z_over_rs + rp_over_rs
    dif_z_rprs = rp_over_rs - z_over_rs
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2
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
    plus_case = torch.cat((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
    minus_case = torch.cat((case3[0], case4[0], case5[0], case6[0], case7[0]))
    star_case = torch.cat((case5[0], case6[0], case7[0], casea[0], casec[0]))

    # cross points
    ph = torch.acos(torch.clamp((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), min=-(1 - EPS), max=1 - EPS))
    theta_1 = torch.zeros_like(z_over_rs)
    ph_case = torch.cat((case5[0], casea[0], casec[0]))
    theta_1[ph_case] = ph[ph_case]
    theta_2 = torch.asin(torch.min(rp_over_rs / z_over_rs, torch.ones_like(z_over_rs)))
    theta_2[case1] = np.pi
    theta_2[case2] = np.pi / 2.0
    theta_2[casea] = np.pi
    theta_2[casec] = np.pi / 2.0
    theta_2[case7] = ph[case7]

    # flux_upper
    plusflux = torch.zeros_like(z_over_rs)

    plusflux[plus_case] = integral_plus_core(method, limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
                                             theta_1[plus_case], theta_2[plus_case], precision=precision)
    if len(case0[0]) > 0:
        plusflux[case0] = integral_centred(method, limb_darkening_coefficients, rp_over_rs, rp_over_rs.new_zeros(1), np.pi)
    if len(caseb[0]) > 0:
        plusflux[caseb] = integral_centred(method, limb_darkening_coefficients, rp_over_rs.new_ones(1), rp_over_rs.new_zeros(1), np.pi)

    # flux_lower
    minsflux = torch.zeros_like(z_over_rs)
    minsflux[minus_case] = integral_minus_core(method, limb_darkening_coefficients, rp_over_rs,
                                               z_over_rs[minus_case], rp_over_rs.new_zeros(1), theta_2[minus_case],
                                               precision=precision)

    # flux_star
    starflux = torch.zeros_like(z_over_rs)
    starflux[star_case] = integral_centred(method, limb_darkening_coefficients, rp_over_rs.new_ones(1), rp_over_rs.new_zeros(1),
                                           ph[star_case])

    # flux_total
    total_flux = integral_centred(method, limb_darkening_coefficients, rp_over_rs.new_ones(1), rp_over_rs.new_zeros(1), 2.0 * np.pi)

    return 1 - (2.0 / total_flux) * (plusflux + starflux - minsflux)


def transit(method, limb_darkening_coefficients, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
            mid_time, time_array, precision=3):
    position_vector = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)

    projected_distance = torch.where(
        position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
        torch.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    return transit_flux_drop(method, limb_darkening_coefficients, rp_over_rs, projected_distance, precision=precision)


def eclipse(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array,
            precision=3):
    position_vector = exoplanet_orbit(period, - sma_over_rs / rp_over_rs, eccentricity, inclination, periastron,
                                      mid_time, time_array)

    projected_distance = torch.where(
        position_vector[0] < 0, 1.0 + 5.0 / rp_over_rs,
        torch.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    return (1.0 + fp_over_fs * transit_flux_drop('claret', [0, 0, 0, 0], 1 / rp_over_rs, projected_distance,
                                                 precision=precision)) / (1.0 + fp_over_fs)


def transit_integrated(method, limb_darkening_coefficients, rp_over_rs,
                       period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array,
                       exp_time, time_factor, precision=3):
    raise NotImplementedError
