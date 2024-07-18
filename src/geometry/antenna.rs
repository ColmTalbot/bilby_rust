use std::f64::consts::FRAC_PI_2;

use numpy::{PyArray1, PyArray2};
use physical_constants::SPEED_OF_LIGHT_IN_VACUUM;
use pyo3::{exceptions::PyValueError, pyfunction, Py, PyErr, PyResult, Python};

use super::{
    polarization::polarization_tensor,
    ra_dec_to_theta_phi,
    util::{SphericalAngles, ThreeVector},
};

const GEOCENTER: ThreeVector = ThreeVector {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

#[allow(dead_code)]
#[pyfunction]
pub fn get_polarization_tensor(
    ra: f64,
    dec: f64,
    gps_time: f64,
    psi: f64,
    mode: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let result =
        std::panic::catch_unwind(|| polarization_tensor(ra, dec, gps_time, psi, mode).into());

    match result {
        Ok(array) => Ok(array),
        Err(_) => Err(PyErr::new::<PyValueError, _>(
            "Error in get_polarization_tensor: {mode} not a polarization mode!",
        )),
    }
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_delay_geocentric(
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: f64,
) -> f64 {
    _time_delay_from_vertices(vertex_1.into(), vertex_2.into(), ra, dec, gps_time)
}

fn _time_delay_from_vertices(
    vertex_1: ThreeVector,
    vertex_2: ThreeVector,
    ra: f64,
    dec: f64,
    gps_time: f64,
) -> f64 {
    let theta_phi: SphericalAngles = ra_dec_to_theta_phi(ra, dec, gps_time).into();

    (vertex_2 - vertex_1).dot(theta_phi.into()) / SPEED_OF_LIGHT_IN_VACUUM
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_delay_from_geocenter(vertex: [f64; 3], ra: f64, dec: f64, gps_time: f64) -> f64 {
    _time_delay_from_vertices(vertex.into(), GEOCENTER, ra, dec, gps_time)
}

#[allow(dead_code)]
#[pyfunction]
pub fn calculate_arm(
    arm_tilt: f64,
    arm_azimuth: f64,
    longitude: f64,
    latitude: f64,
) -> Py<PyArray1<f64>> {
    let vec1: ThreeVector = SphericalAngles {
        zenith: -latitude,
        azimuth: longitude,
    }
    .into();
    let vec2: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2,
        azimuth: FRAC_PI_2 + longitude,
    }
    .into();
    let vec3: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2 - latitude,
        azimuth: longitude,
    }
    .into();
    (vec1 * arm_tilt.cos() * arm_azimuth.sin()
        + vec2 * arm_tilt.cos() * arm_azimuth.cos()
        + vec3 * arm_tilt.sin())
    .into()
}

#[allow(dead_code)]
#[pyfunction]
pub fn detector_tensor(x: [f64; 3], y: [f64; 3]) -> Py<PyArray2<f64>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();

    ((x.outer(x) - y.outer(y)) / 2.0).into()
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_delay_from_geocenter_vectorized(
    vertex: [f64; 3],
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
) -> Py<PyArray1<f64>> {
    let times = gps_times
        .iter()
        .map(|&gps_time| time_delay_from_geocenter(vertex, ra, dec, gps_time))
        .collect();
    Python::with_gil(|py| PyArray1::from_vec_bound(py, times).unbind())
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_dependent_polarization_tensor(
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
    psi: f64,
    mode: &str,
) -> Vec<Vec<Vec<f64>>> {
    let mut output: Vec<Vec<Vec<f64>>> = Vec::new();
    for gps_time in gps_times {
        let pol = polarization_tensor(ra, dec, gps_time, psi, mode);
        output.push(pol.into());
    }
    output
}
