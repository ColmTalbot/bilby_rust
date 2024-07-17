use pyo3::{Py, pyfunction};
use numpy::PyArray2;

use crate::geometry::util;

pub fn _rotation_matrix_from_vertices(vertex_1: &[f64; 3], vertex_2: &[f64; 3]) -> [[f64; 3]; 3] {
    let delta_x = util::normalized(&[
        vertex_1[0] - vertex_2[0],
        vertex_1[1] - vertex_2[1],
        vertex_1[2] - vertex_2[2],
    ]);
    let midpoint = util::normalized(&[
        (vertex_1[0] + vertex_2[0]),
        (vertex_1[1] + vertex_2[1]),
        (vertex_1[2] + vertex_2[2]),
    ]);
    let x_axis = util::cross_product(&delta_x, &midpoint);
    let y_axis = util::cross_product(&x_axis, &delta_x);
    [x_axis, y_axis, delta_x]
}

pub fn _rotation_matrix_from_delta_x(delta_x: &[f64; 3]) -> [[f64; 3]; 3] {
    let delta_x = util::normalized(delta_x);
    
    let beta = delta_x[2].acos();
    let alpha = (-delta_x[1] * delta_x[2]).atan2(delta_x[0]);
    let gamma = delta_x[1].atan2(delta_x[0]);

    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();
    let cos_beta = beta.cos();
    let sin_beta = beta.sin();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();

    [
        [
            cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma,
            -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma,
            sin_beta * cos_gamma,
        ],
        [
            cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma,
            -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma,
            sin_beta * sin_gamma,
        ],
        [
            -cos_alpha * sin_beta,
            sin_alpha * sin_beta,
            cos_beta,
        ]
    ]
}

pub fn rotate_spherical_angles(zenith: f64, azimuth: f64, rotation: [[f64; 3]; 3]) -> (f64, f64) {
    let cazimuth = azimuth.cos();
    let sazimuth = azimuth.sin();
    let czenith = zenith.cos();
    let szenith = zenith.sin();

    let theta = (
        rotation[2][0] * szenith * cazimuth
        + rotation[2][1]* szenith * sazimuth
        + rotation[2][2] * czenith
    ).acos();
    let phi = (
        rotation[1][0] * cazimuth * szenith
        + rotation[1][1] * sazimuth * szenith
        + rotation[1][2] * czenith
    ).atan2(
        rotation[0][0] * cazimuth * szenith
        + rotation[0][1] * sazimuth * szenith
        + rotation[0][2] * czenith
    );
    (theta, phi)
}

#[allow(dead_code)]
#[pyfunction]
pub fn rotation_matrix_from_vertices(vertex_1: [f64; 3], vertex_2: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = _rotation_matrix_from_vertices(&vertex_1, &vertex_2);
    util::_wrap_2d_array_for_numpy(&rotation)
}

#[allow(dead_code)]
#[pyfunction]
pub fn rotation_matrix_from_delta_x(delta_x: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = _rotation_matrix_from_delta_x(&delta_x);
    util::_wrap_2d_array_for_numpy(&rotation)
}

#[allow(dead_code)]
#[pyfunction]
pub fn zenith_azimuth_to_theta_phi(zenith: f64, azimuth: f64, delta_x: [f64; 3]) -> (f64, f64) {
    let rotation = _rotation_matrix_from_delta_x(&delta_x);
    rotate_spherical_angles(zenith, azimuth, rotation)
}

#[allow(dead_code)]
#[pyfunction]
pub fn zenith_azimuth_to_theta_phi_optimized(
    zenith: f64, azimuth: f64, vertex_1:[f64; 3], vertex_2: [f64; 3]
) -> (f64, f64) {
    let rotation = _rotation_matrix_from_vertices(&vertex_1, &vertex_2);
    rotate_spherical_angles(zenith, azimuth, rotation)
}
