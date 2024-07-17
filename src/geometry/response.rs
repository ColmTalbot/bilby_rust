use std::f64::consts::PI;

use pyo3::{pyfunction, Py, Python};
use numpy::{Complex64, PyArray1, PyArray2, PyArray3};
use num_complex::Complex;

use crate::geometry::util::{inner_product, outer, ra_dec_to_theta_phi};

use super::polarization::polarization_tensor;

fn projection(
    frequency: &f64, cos_angle: f64, free_spectral_range: f64
) -> Complex<f64> {
    let omega = Complex::I * PI * frequency / free_spectral_range;
    1.0 / (4.0 * omega) * (
        (1.0 - (-(1.0 - cos_angle) * omega).exp()) / (1.0 - cos_angle)
        - (-2.0 * omega).exp() * (1.0 - ((1.0 + cos_angle) * omega).exp()) / (1.0 + cos_angle)
    )
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

    let x_tensor = outer(&x, &x);
    let y_tensor = outer(&y, &y);

    let mut output: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
    for (frequency, gps_time) in frequencies.iter().zip(gps_times.iter()) {
        let temp: [[Complex<f64>; 3]; 3] = _single_finite_size_detector_tensor(
            frequency, gps_time, &x, &y, &x_tensor, &y_tensor, ra, dec, free_spectral_range
        );
        output.push(temp.iter().map(|row| row.to_vec()).collect());
    }
    Python::with_gil(|py| {
        PyArray3::from_vec3_bound(py, &output).unwrap().unbind()
    })
}

fn _single_finite_size_detector_tensor(
    frequency: &f64,
    gps_time: &f64,
    x: &[f64; 3],
    y: &[f64; 3],
    x_tensor: &[[f64; 3]; 3],
    y_tensor: &[[f64; 3]; 3],
    ra: f64,
    dec: f64,
    free_spectral_range: f64,
) -> [[Complex<f64>; 3]; 3] {
    let line_of_sight = line_of_sight(ra, dec, *gps_time);
    let cos_xangle = inner_product(x, &line_of_sight);
    let cos_yangle = inner_product(y, &line_of_sight);
    let delta_x = projection(frequency, cos_xangle, free_spectral_range);
    let delta_y = projection(frequency, cos_yangle, free_spectral_range);

    let mut temp: [[Complex<f64>; 3]; 3] = [[Complex::new(0.0, 0.0); 3]; 3];
    temp.iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, element)| {
            *element = x_tensor[i][j] * delta_x - y_tensor[i][j] * delta_y;
        });
    });
    temp
}

fn line_of_sight(ra: f64, dec: f64, gps_time: f64) -> [f64; 3] {
    let theta_phi = ra_dec_to_theta_phi(ra, dec, gps_time);
    let theta = theta_phi.0;
    let phi = theta_phi.1;
    [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ]
}

#[allow(dead_code)]
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
    let x_tensor = outer(&x, &x);
    let y_tensor = outer(&y, &y);

    let mut output: Vec<Complex<f64>> = Vec::new();

    for (frequency, gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pol = polarization_tensor(ra, dec, *gps_time, psi, mode);

        let det: [[Complex<f64>; 3]; 3] = _single_finite_size_detector_tensor(
            frequency, gps_time, &x, &y, &x_tensor, &y_tensor, ra, dec, free_spectral_range
        );

        let mut temp: Complex<f64> = Complex::new(0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                temp += det[i][j] * pol[i][j];
            }
        }
        output.push(temp);
    }
    Python::with_gil(|py| {PyArray1::from_vec_bound(py, output).unbind()})
}

#[allow(dead_code)]
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
    let x_tensor = outer(&x, &x);
    let y_tensor = outer(&y, &y);

    let mut output: Vec<Vec<Complex<f64>>> = Vec::new();

    for (frequency, gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pols: [[[f64; 3]; 3]; 2] = [
            polarization_tensor(ra, dec, *gps_time, psi, "plus"),
            polarization_tensor(ra, dec, *gps_time, psi, "cross"),
        ];

        let det: [[Complex<f64>; 3]; 3] = _single_finite_size_detector_tensor(
            frequency, gps_time, &x, &y, &x_tensor, &y_tensor, ra, dec, free_spectral_range
        );

        let mut temp: [Complex<f64>; 2] = [Complex::new(0.0, 0.0); 2];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..3 {
                    temp[i] += det[j][k] * pols[i][j][k];
                }
            }
        }
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| {
        PyArray2::from_vec2_bound(py, &output).unwrap().unbind()
    })
}

#[allow(dead_code)]
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
    let x_tensor = outer(&x, &x);
    let y_tensor = outer(&y, &y);

    let mut output: Vec<Vec<Complex<f64>>> = Vec::new();

    for (frequency, gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pols: [[[f64; 3]; 3]; 6] = [
            polarization_tensor(ra, dec, *gps_time, psi, "plus"),
            polarization_tensor(ra, dec, *gps_time, psi, "cross"),
            polarization_tensor(ra, dec, *gps_time, psi, "breathing"),
            polarization_tensor(ra, dec, *gps_time, psi, "longitudinal"),
            polarization_tensor(ra, dec, *gps_time, psi, "x"),
            polarization_tensor(ra, dec, *gps_time, psi, "y"),
        ];

        let det: [[Complex<f64>; 3]; 3] = _single_finite_size_detector_tensor(
            frequency, gps_time, &x, &y, &x_tensor, &y_tensor, ra, dec, free_spectral_range
        );

        let mut temp: [Complex<f64>; 6] = [Complex::new(0.0, 0.0); 6];
        for i in 0..6 {
            for j in 0..3 {
                for k in 0..3 {
                    temp[i] += det[j][k] * pols[i][j][k];
                }
            }
        }
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| {
        PyArray2::from_vec2_bound(py, &output).unwrap().unbind()
    })
}
