use std::f64::consts::PI;

use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArray2, PyArray3};
use pyo3::{pyfunction, Py, Python};

use super::{
    polarization::polarization_tensor,
    ra_dec_to_theta_phi,
    util::{ComplexThreeMatrix, SphericalAngles, ThreeMatrix, ThreeVector},
};

fn projection(frequency: &f64, cos_angle: f64, free_spectral_range: f64) -> Complex<f64> {
    let omega = Complex::I * PI * frequency / free_spectral_range;
    1.0 / (4.0 * omega)
        * ((1.0 - (-(1.0 - cos_angle) * omega).exp()) / (1.0 - cos_angle)
            - (-2.0 * omega).exp() * (1.0 - ((1.0 + cos_angle) * omega).exp()) / (1.0 + cos_angle))
}

#[allow(dead_code)]
#[pyfunction]
pub fn frequency_dependent_detector_tensor(
    x: [f64; 3],
    y: [f64; 3],
    frequencies: Vec<f64>,
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray3<Complex64>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();
    let x_tensor = x.outer(x);
    let y_tensor = y.outer(y);

    let mut output: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
    for (frequency, gps_time) in frequencies.iter().zip(gps_times.iter()) {
        let temp: ComplexThreeMatrix = _single_finite_size_detector_tensor(
            frequency,
            gps_time,
            &x,
            &y,
            &x_tensor,
            &y_tensor,
            ra,
            dec,
            free_spectral_range,
        );
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| PyArray3::from_vec3_bound(py, &output).unwrap().unbind())
}

#[allow(clippy::too_many_arguments)]
fn _single_finite_size_detector_tensor(
    frequency: &f64,
    gps_time: &f64,
    x: &ThreeVector,
    y: &ThreeVector,
    x_tensor: &ThreeMatrix,
    y_tensor: &ThreeMatrix,
    ra: f64,
    dec: f64,
    free_spectral_range: f64,
) -> ComplexThreeMatrix {
    let line_of_sight = line_of_sight(ra, dec, *gps_time);
    let cos_xangle = x.dot(line_of_sight);
    let cos_yangle = y.dot(line_of_sight);
    let delta_x = projection(frequency, cos_xangle, free_spectral_range);
    let delta_y = projection(frequency, cos_yangle, free_spectral_range);

    x_tensor * delta_x - y_tensor * delta_y
}

fn line_of_sight(ra: f64, dec: f64, gps_time: f64) -> ThreeVector {
    let theta_phi: SphericalAngles = ra_dec_to_theta_phi(ra, dec, gps_time).into();
    theta_phi.into()
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    mode: &str,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray1<Complex<f64>>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();
    let x_tensor = x.outer(x);
    let y_tensor = y.outer(y);

    let mut output: Vec<Complex<f64>> = Vec::new();

    for (frequency, gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pol: ThreeMatrix = polarization_tensor(ra, dec, *gps_time, psi, mode);

        let det: ComplexThreeMatrix = _single_finite_size_detector_tensor(
            frequency,
            gps_time,
            &x,
            &y,
            &x_tensor,
            &y_tensor,
            ra,
            dec,
            free_spectral_range,
        );

        let mut temp: Complex<f64> = Complex::new(0.0, 0.0);
        for i in 0..3 {
            temp += det.rows[i].x * pol.rows[i].x;
            temp += det.rows[i].y * pol.rows[i].y;
            temp += det.rows[i].z * pol.rows[i].z;
        }
        output.push(temp);
    }
    Python::with_gil(|py| PyArray1::from_vec_bound(py, output).unbind())
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response_tensor_modes(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray2<Complex<f64>>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();
    let x_tensor = x.outer(x);
    let y_tensor = y.outer(y);

    let mut output: Vec<Vec<Complex<f64>>> = Vec::new();

    for (frequency, gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pols: [ThreeMatrix; 2] = [
            polarization_tensor(ra, dec, *gps_time, psi, "plus"),
            polarization_tensor(ra, dec, *gps_time, psi, "cross"),
        ];

        let det: ComplexThreeMatrix = _single_finite_size_detector_tensor(
            frequency,
            gps_time,
            &x,
            &y,
            &x_tensor,
            &y_tensor,
            ra,
            dec,
            free_spectral_range,
        );

        let mut temp: [Complex<f64>; 2] = [Complex::new(0.0, 0.0); 2];
        for i in 0..2 {
            temp[i] = (det * pols[i]).sum();
        }
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| PyArray2::from_vec2_bound(py, &output).unwrap().unbind())
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response_all_modes(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray2<Complex<f64>>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();
    let x_tensor = x.outer(x);
    let y_tensor = y.outer(y);

    let mut output: Vec<Vec<Complex<f64>>> = Vec::new();

    for (frequency, gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pols: [ThreeMatrix; 6] = [
            polarization_tensor(ra, dec, *gps_time, psi, "plus"),
            polarization_tensor(ra, dec, *gps_time, psi, "cross"),
            polarization_tensor(ra, dec, *gps_time, psi, "breathing"),
            polarization_tensor(ra, dec, *gps_time, psi, "longitudinal"),
            polarization_tensor(ra, dec, *gps_time, psi, "x"),
            polarization_tensor(ra, dec, *gps_time, psi, "y"),
        ];

        let det: ComplexThreeMatrix = _single_finite_size_detector_tensor(
            frequency,
            gps_time,
            &x,
            &y,
            &x_tensor,
            &y_tensor,
            ra,
            dec,
            free_spectral_range,
        );

        let mut temp: [Complex<f64>; 6] = [Complex::new(0.0, 0.0); 6];
        for i in 0..6 {
            temp[i] = (det * pols[i]).sum();
        }
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| PyArray2::from_vec2_bound(py, &output).unwrap().unbind())
}
