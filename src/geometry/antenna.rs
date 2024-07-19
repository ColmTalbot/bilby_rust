use std::f64::consts::{FRAC_PI_2, PI};

use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArray3};
use physical_constants::SPEED_OF_LIGHT_IN_VACUUM;
use pyo3::{exceptions::PyValueError, pyfunction, Py, PyErr, PyResult, Python};

use super::{
    polarization::polarization_tensor,
    three::{ComplexThreeMatrix, SphericalAngles, ThreeMatrix, ThreeVector},
    util::{line_of_sight, ra_dec_to_theta_phi},
};

pub struct DetectorGeometry {
    x: ThreeVector,
    y: ThreeVector,
    free_spectral_range: f64,
    x_tensor: ThreeMatrix,
    y_tensor: ThreeMatrix,
    detector_tensor: ThreeMatrix,
}

impl DetectorGeometry {
    pub fn new(x: ThreeVector, y: ThreeVector, free_spectral_range: f64) -> Self {
        let x_tensor = x.outer(x);
        let y_tensor = y.outer(y);
        Self {
            x,
            y,
            free_spectral_range,
            x_tensor,
            y_tensor,
            detector_tensor: (x_tensor - y_tensor) / 2.0,
        }
    }

    pub fn finite_size_tensor(
        &self,
        frequency: f64,
        gps_time: f64,
        ra: f64,
        dec: f64,
    ) -> ComplexThreeMatrix {
        let line_of_sight = line_of_sight(ra, dec, gps_time);
        let cos_xangle = self.x.dot(line_of_sight);
        let cos_yangle = self.y.dot(line_of_sight);
        let delta_x = projection(frequency, cos_xangle, self.free_spectral_range);
        let delta_y = projection(frequency, cos_yangle, self.free_spectral_range);

        self.x_tensor * delta_x - self.y_tensor * delta_y
    }
}

fn projection(frequency: f64, cos_angle: f64, free_spectral_range: f64) -> Complex<f64> {
    let omega = Complex::I * PI * frequency / free_spectral_range;
    1.0 / (4.0 * omega)
        * ((1.0 - (-(1.0 - cos_angle) * omega).exp()) / (1.0 - cos_angle)
            - (-2.0 * omega).exp() * (1.0 - ((1.0 + cos_angle) * omega).exp()) / (1.0 + cos_angle))
}

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
    let arm_vector: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2 - arm_tilt,
        azimuth: FRAC_PI_2 - arm_azimuth,
    }
    .into();
    ThreeMatrix::from_columns([vec1, vec2, vec3])
        .dot(arm_vector)
        .into()
}

#[allow(dead_code)]
#[pyfunction]
pub fn detector_tensor(x: [f64; 3], y: [f64; 3]) -> Py<PyArray2<f64>> {
    let det = DetectorGeometry::new(x.into(), y.into(), 1.0);
    det.detector_tensor.into()
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
) -> Py<PyArray3<f64>> {
    let output: Vec<Vec<Vec<f64>>> = gps_times
        .iter()
        .map(|&gps_time| polarization_tensor(ra, dec, gps_time, psi, mode).into())
        .collect();
    Python::with_gil(|py| PyArray3::from_vec3_bound(py, &output).unwrap().unbind())
}
