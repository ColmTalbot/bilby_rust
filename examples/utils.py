import numpy as np
from bilby.core.utils.conversion import ra_dec_to_theta_phi
from bilby_cython.geometry import get_polarization_tensor
from bilby_cython.time import greenwich_mean_sidereal_time


def p_frequency_dependent_response(frequency, n_dot_e, free_spectral_range):
    phase = np.pi * frequency / free_spectral_range
    return (
        (
            np.nan_to_num(
                (1 - np.exp(-1j * phase * (1 - n_dot_e)))
                / (1 - n_dot_e)
            ) - np.exp(-2j * phase)
            * np.nan_to_num(
                (1 - np.exp(1j * phase * (1 + n_dot_e)))
                / (1 + n_dot_e)
            )
        ) / (4j * phase)
    )


def detector_tensor(x, y, frequency, free_spectral_range, ra, dec, gmst):
    """
    Calculate the detector tensor for a given pair of arms

    Returns
    =======
    array_like: A 3x3 representation of the detector tensor

    """
    los = line_of_sight(ra, dec, gmst).T
    return (
        p_frequency_dependent_response(frequency, los.dot(x), free_spectral_range)[:, None, None] * np.outer(x, x)
        - p_frequency_dependent_response(frequency, los.dot(y), free_spectral_range)[:, None, None] * np.outer(y, y)
    ).squeeze()


def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

def line_of_sight(ra, dec, gmst):
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    return spherical_to_cartesian(theta * np.ones(gmst.shape), phi * np.ones(gmst.shape))


def antenna_response(x, y, ra, dec, gps_time, psi, mode, frequency, free_spectral_range):
    gmst = greenwich_mean_sidereal_time(gps_time * np.ones(frequency.shape))
    return np.einsum(
        "...jk,jk->...",
        detector_tensor(
            x=x,
            y=y,
            frequency=np.atleast_1d(frequency),
            free_spectral_range=free_spectral_range,
            ra=ra,
            dec=dec,
            gmst=gmst,
        ),
        get_polarization_tensor(ra=ra, dec=dec, time=gps_time, psi=psi, mode=mode)
    )