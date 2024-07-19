import numpy as np
import bilby_rust.geometry as rg
import bilby_cython.geometry as cg
import utils
import pytest


def test_calculate_arm():
    np.testing.assert_allclose(
        cg.calculate_arm(1.0, 1.0, 1.0, 1.0),
        rg.calculate_arm(1.0, 1.0, 1.0, 1.0),
    )


def test_detector_tensor():
    np.testing.assert_allclose(
        cg.detector_tensor(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        rg.detector_tensor(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
    )


def test_get_polarization_tensor():
    np.testing.assert_allclose(
        cg.get_polarization_tensor(1.0, 1.0, 1.0, 1.0, "cross"),
        rg.get_polarization_tensor(1.0, 1.0, 1.0, 1.0, "cross"),
    )


def test_rotation_matrix_from_delta():
    """
    One of the elements is zero and so we need to set the absolute tolerance
    as differences are in units of machine precision.
    """
    np.testing.assert_allclose(
        cg.rotation_matrix_from_delta(np.array([1.0, 2.0, 3.0])),
        rg.rotation_matrix_from_delta_x(np.array([1.0, 2.0, 3.0])),
        atol=1e-14,
    )


def time_delay_from_geocenter():
    np.testing.assert_almost_equal(
        cg.time_delay_from_geocenter(np.array([1.0, 2.0, 3.0]) * 1e9, 1.0, 1.0, 1.0),
        rg.time_delay_from_geocenter(np.array([1.0, 2.0, 3.0]) * 1e9, 1.0, 1.0, 1.0),
    )


def test_zenith_azimuth_to_theta_phi():
    np.testing.assert_almost_equal(
        rg.zenith_azimuth_to_theta_phi(1.0, 1.0, np.array([1.0, 2.0, 3.0])),
        cg.zenith_azimuth_to_theta_phi(1.0, 1.0, np.array([1.0, 2.0, 3.0])),
    )


@pytest.mark.parametrize("mode", ["plus", "cross", "longitudinal", "breathing", "x", "y"])
def test_antenna_response(mode):
    np.testing.assert_almost_equal(
        rg.antenna_response(
            x=np.array([-0.22389266,  0.79983063,  0.55690488]),
            y=np.array([-0.91397819,  0.02609404, -0.40492342]),
            frequency=np.linspace(10, 4096, 1000),
            ra=1.0,
            dec=1.0,
            gps_time=np.zeros(1000),
            free_spectral_range=2000,
            psi=1.0,
            mode=mode,
        ),
        utils.antenna_response(
            x=np.array([-0.22389266,  0.79983063,  0.55690488]),
            y=np.array([-0.91397819,  0.02609404, -0.40492342]),
            frequency=np.linspace(10, 4096, 1000),
            free_spectral_range=2000.0,
            ra=1.0,
            dec=1.0,
            gps_time=0.0,
            psi=1.0,
            mode=mode,
        )
    )