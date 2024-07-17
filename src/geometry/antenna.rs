use physical_constants::SPEED_OF_LIGHT_IN_VACUUM;
use pyo3::{PyErr, exceptions::PyValueError, pyfunction, Py, PyResult, Python};
use numpy::{PyArray1, PyArray2};

use crate::geometry::polarization::polarization_tensor;
use crate::geometry::util::{
    _wrap_2d_array_for_numpy,
    add_three_by_three,
    outer,
    ra_dec_to_theta_phi,
};

const GEOCENTER: [f64; 3] = [0.0, 0.0, 0.0];

#[allow(dead_code)]
#[pyfunction]
pub fn get_polarization_tensor(
    ra: f64, dec: f64, gps_time: f64, psi: f64, mode: &str
) -> PyResult<Py<PyArray2<f64>>> {
    let result = std::panic::catch_unwind(
        || { 
            let pol = polarization_tensor(ra, dec, gps_time, psi, mode);
            _wrap_2d_array_for_numpy(&pol)
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

    (
        (vertex_2[0] - vertex_1[0]) * theta.sin() * phi.cos()
        + (vertex_2[1] - vertex_1[1]) * theta.sin() * phi.cos()
        + (vertex_2[2] - vertex_1[2]) * theta.cos()
    ) / SPEED_OF_LIGHT_IN_VACUUM
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
    let arm = [
        - longitude.sin() * arm_tilt.cos() * arm_azimuth.cos()
        - latitude.sin() * longitude.cos() * arm_tilt.cos() * arm_azimuth.sin()
        + latitude.cos() * longitude.cos() * arm_tilt.sin(),
        longitude.cos() * arm_tilt.cos() * arm_azimuth.cos()
        - latitude.sin() * longitude.sin() * arm_tilt.cos() * arm_azimuth.sin()
        + latitude.cos() * longitude.sin() * arm_tilt.sin(),
        latitude.cos() * arm_tilt.cos() * arm_azimuth.sin()
        + latitude.sin() * arm_tilt.sin(),
    ];
    Python::with_gil(|py| {
        PyArray1::from_vec_bound(py, arm.to_vec()).unbind()
    })
}

#[allow(dead_code)]
#[pyfunction]
pub fn detector_tensor(x: [f64; 3], y: [f64; 3]) -> Py<PyArray2<f64>> {
    let mut output = add_three_by_three(
        &outer(&x, &x),
        &outer(&y, &y),
        &std::ops::Sub::sub,
    );
    for row in output.iter_mut() {
        for element in row.iter_mut() {
            *element /= 2.0;
        }
    }
    _wrap_2d_array_for_numpy(&output)
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
        output.push(pol.iter().map(|row| row.to_vec()).collect());
    }
    output
}
