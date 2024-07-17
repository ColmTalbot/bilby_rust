use std::f64::consts::PI;

use pyo3::{pyfunction, Py, Python};
use numpy::{Complex64, PyArray3};
use num_complex::Complex;

use crate::geometry::util::{outer, ra_dec_to_theta_phi};

fn projection_factor(
    frequencies: &Vec<f64>, arm: &[f64; 3], line_of_sight: &Vec<[f64; 3]>, free_spectral_range: f64
) -> Vec<Complex<f64>> {
    let mut output: Vec<Complex<f64>> = Vec::new();
    for (f, n) in frequencies.iter().zip(line_of_sight.iter()) {
        let cos_angle = arm.iter().zip(n.iter()).map(|(a, b)| a * b).sum::<f64>();
        output.push(_projection(f, cos_angle, free_spectral_range));
    }
    output
}

fn _projection(
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

    let line_of_sight = gps_times.iter().map(
        |&gps_time| line_of_sight(ra, dec, gps_time)
    ).collect();

    let x_tensor = outer(&x, &x);
    let y_tensor = outer(&y, &y);

    let delta_xarm = projection_factor(
        &frequencies, &x, &line_of_sight, free_spectral_range
    );
    let delta_yarm = projection_factor(
        &frequencies, &y, &line_of_sight, free_spectral_range
    );

    let mut output: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
    for (delta_x, delta_y) in delta_xarm.iter().zip(delta_yarm.iter()) {
        let mut temp: [[Complex<f64>; 3]; 3] = [[Complex::new(0.0, 0.0); 3]; 3];
        temp.iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(j, element)| {
                *element = x_tensor[i][j] * delta_x - y_tensor[i][j] * delta_y;
            });
        });
        output.push(temp.iter().map(|row| row.to_vec()).collect());
    }
    Python::with_gil(|py| {
        PyArray3::from_vec3_bound(py, &output).unwrap().unbind()
    })
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

