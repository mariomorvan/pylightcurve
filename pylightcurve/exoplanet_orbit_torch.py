from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from ._1databases import *

torch.set_default_dtype(torch.float64)


def exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array,
                    ww=torch.zeros(1)):
    if torch.isnan(periastron):
        periastron = torch.zeros(1)
    inclination = inclination * np.pi / 180.0
    periastron = periastron * np.pi / 180.0
    ww = ww * np.pi / 180.0

    if eccentricity == 0 and ww == 0:
        vv = 2 * np.pi * (time_array - mid_time) / period
        bb = sma_over_rs * torch.cos(vv)
        return [bb * torch.sin(inclination), sma_over_rs * torch.sin(vv), - bb * torch.cos(inclination)]

    if periastron < np.pi / 2:
        aa = 1.0 * np.pi / 2 - periastron
    else:
        aa = 5.0 * np.pi / 2 - periastron
    bb = 2 * torch.atan(torch.sqrt((1 - eccentricity) / (1 + eccentricity)) * torch.tan(aa / 2))
    if bb < 0:
        bb += 2 * np.pi
    mid_time = mid_time - (period / 2.0 / np.pi) * (bb - eccentricity * torch.sin(bb))
    m = (time_array - mid_time - ((time_array - mid_time) / period).int() * period) * 2.0 * np.pi / period
    u0 = m
    stop = False
    u1 = 0
    for ii in range(10000):  # setting a limit of 1k iterations - arbitrary limit
        u1 = u0 - (u0 - eccentricity * torch.sin(u0) - m) / (1 - eccentricity * torch.cos(u0))
        stop = (torch.abs(u1 - u0) < 10 ** (-7)).all()
        if stop:
            break
        else:
            u0 = u1.clone()
    if not stop:
        raise RuntimeError('Failed to find a solution in 10000 loops')

    vv = 2 * torch.atan(torch.sqrt((1 + eccentricity) / (1 - eccentricity)) * torch.tan(u1 / 2))
    #
    rr = sma_over_rs * (1 - (eccentricity ** 2)) / (torch.ones_like(vv) + eccentricity * torch.cos(vv))
    aa = torch.cos(vv + periastron)
    bb = torch.sin(vv + periastron)
    x = rr * bb * torch.sin(inclination)
    y = rr * (-aa * torch.cos(ww) + bb * torch.sin(ww) * torch.cos(inclination))
    z = rr * (-aa * torch.sin(ww) - bb * torch.cos(ww) * torch.cos(inclination))

    return [x, y, z]


def transit_projected_distance(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array):
    position_vector = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)

    return (torch.where(position_vector[0] < 0, - torch.ones_like(position_vector[0]),
                        torch.ones_like(position_vector[0])) *
            torch.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))


def transit_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron):
    ww = periastron * np.pi / 180
    ii = inclination * np.pi / 180
    ee = eccentricity
    aa = sma_over_rs
    ro_pt = (1 - ee ** 2) / (1 + ee * torch.sin(ww))
    b_pt = aa * ro_pt * torch.cos(ii)
    if b_pt > 1:
        b_pt = 0.5
    s_ps = 1.0 + rp_over_rs
    df = torch.asin(torch.sqrt((s_ps ** 2 - b_pt ** 2) / ((aa ** 2) * (ro_pt ** 2) - b_pt ** 2)))
    abs_value = (period * (ro_pt ** 2)) / (np.pi * torch.sqrt(1 - ee ** 2)) * df

    return abs_value
