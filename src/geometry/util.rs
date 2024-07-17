use std::f64::consts::PI;

use pyo3::{Py, Python, pyfunction};
use numpy::PyArray2;

use crate::time;

#[pyfunction]
pub fn ra_dec_to_theta_phi(ra: f64, dec: f64, gps_time: f64) -> (f64, f64) {
    let gmst = time::greenwich_mean_sidereal_time(gps_time) % (2.0 * PI);
    let theta = PI / 2.0 - dec;
    let phi = ra - gmst;
    (theta, phi)
}

pub fn cross_product(vertex_1: &[f64; 3], vertex_2: &[f64; 3]) -> [f64; 3] {
    [
        vertex_1[1] * vertex_2[2] - vertex_1[2] * vertex_2[1],
        vertex_1[2] * vertex_2[0] - vertex_1[0] * vertex_2[2],
        vertex_1[0] * vertex_2[1] - vertex_1[1] * vertex_2[0],
    ]
}

pub fn normalized(v: &[f64; 3]) -> [f64; 3] {
    let norm = norm(v);
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

fn norm(v: &[f64; 3]) -> f64 {
    (v.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt()
}

pub fn outer(v: &[f64; 3], w: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut output = [[0.0; 3]; 3];
    output.iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, element)| {
            *element = v[i] * w[j];
        });
    });
    output
}

pub fn add_three_by_three(v: &[[f64; 3]; 3], w: &[[f64; 3]; 3], op: &dyn Fn(f64, f64) -> f64) -> [[f64; 3]; 3] {
    let mut output = [[0.0; 3]; 3];
    output.iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, element)| {
            *element = op(v[i][j], w[i][j]);
        });
    });
    output
}

pub fn _wrap_2d_array_for_numpy(array: &[[f64; 3]; 3]) -> Py<PyArray2<f64>> {
    let output: Vec<Vec<f64>> = array.iter().map(|row| row.to_vec()).collect();
    Python::with_gil(|py| {
        PyArray2::from_vec2_bound(py, &output).unwrap().unbind()
    })
}