use std::f64::consts::FRAC_PI_2;

use physical_constants::SPEED_OF_LIGHT_IN_VACUUM;
use pyo3::{PyErr, exceptions::PyValueError, pyfunction, Py, PyResult, Python};
use numpy::{PyArray1, PyArray2};

use super::polarization::polarization_tensor;
use super::util::{ra_dec_to_theta_phi, ThreeVector};

const GEOCENTER: [f64; 3] = [0.0, 0.0, 0.0];

#[allow(dead_code)]
#[pyfunction]
pub fn get_polarization_tensor(
    ra: f64, dec: f64, gps_time: f64, psi: f64, mode: &str
) -> PyResult<Py<PyArray2<f64>>> {
    let result = std::panic::catch_unwind(
        || { 
            let pol = polarization_tensor(ra, dec, gps_time, psi, mode);
            pol.to_pyarray()
        }
    );

    match result {
        Ok(array) => Ok(array),
        Err(_) => Err(PyErr::new::<PyValueError, _>(
            "Error in get_polarization_tensor: {mode} not a polarization mode!"
        )),
    }
}    

#[allow(dead_code)]
#[pyfunction]
pub fn time_delay_geocentric(
    vertex_1: [f64; 3], vertex_2: [f64; 3], ra: f64, dec: f64, gps_time: f64
) -> f64 {
    let theta_phi = ra_dec_to_theta_phi(ra, dec, gps_time);
    let theta = theta_phi.0;
    let phi = theta_phi.1;

    let vertex_1 = ThreeVector::from_array(&vertex_1);
    let vertex_2 = ThreeVector::from_array(&vertex_2);
    let vector = ThreeVector::from_spherical_angles(theta, phi);

    (vertex_2 - vertex_1).dot(&vector) / SPEED_OF_LIGHT_IN_VACUUM
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_delay_from_geocenter(vertex: [f64; 3], ra: f64, dec: f64, gps_time: f64) -> f64 {
    time_delay_geocentric(vertex, GEOCENTER, ra, dec, gps_time)
}

#[allow(dead_code)]
#[pyfunction]
pub fn calculate_arm(
    arm_tilt: f64, arm_azimuth: f64, longitude: f64, latitude: f64
) -> Py<PyArray1<f64>> {
    let output = ThreeVector::from_spherical_angles(-latitude, longitude) * arm_tilt.cos() * arm_azimuth.sin()
        + ThreeVector::from_spherical_angles(FRAC_PI_2, FRAC_PI_2 + longitude) * arm_tilt.cos() * arm_azimuth.cos()
        + ThreeVector::from_spherical_angles(FRAC_PI_2 - latitude, longitude) * arm_tilt.sin();
    output.to_pyarray()
}

#[allow(dead_code)]
#[pyfunction]
pub fn detector_tensor(x: [f64; 3], y: [f64; 3]) -> Py<PyArray2<f64>> {
    let x = ThreeVector::from_array(&x);
    let y = ThreeVector::from_array(&y);

    ((x.outer(&x) - y.outer(&y)) / 2.0).to_pyarray()
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_delay_from_geocenter_vectorized(
    vertex: [f64; 3], ra: f64, dec: f64, gps_times: Vec<f64>
) -> Py<PyArray1<f64>> {
    let times = gps_times.iter().map(
        |&gps_time| time_delay_from_geocenter(vertex, ra, dec, gps_time)
    ).collect();
    Python::with_gil(|py| {PyArray1::from_vec_bound(py, times).unbind()})
}

#[allow(dead_code)]
#[pyfunction]
pub fn time_dependent_polarization_tensor(
    ra: f64, dec: f64, gps_times: Vec<f64>, psi: f64, mode: &str
) -> Vec<Vec<Vec<f64>>> {
    let mut output: Vec<Vec<Vec<f64>>> = Vec::new();
    for gps_time in gps_times {
        let pol = polarization_tensor(ra, dec, gps_time, psi, mode);
        output.push(pol.to_vec());
    }
    output
}
