use std::f64::consts::{PI, FRAC_PI_2};

use pyo3::{Py, pyfunction};
use numpy::PyArray2;

use super::util::{ThreeMatrix, ThreeVector};


pub fn _rotation_matrix_from_vertices(vertex_1: ThreeVector, vertex_2: ThreeVector) -> ThreeMatrix {
    let delta_x = (vertex_1 - vertex_2).normalize();
    let midpoint = (vertex_1 + vertex_2).normalize();
    let x_axis = delta_x.cross(&midpoint);
    let y_axis = x_axis.cross(&delta_x);
    ThreeMatrix {
        rows: [
            x_axis,
            y_axis,
            delta_x,
        ]
    }
}

pub fn _rotation_matrix_from_delta_x(delta_x: ThreeVector) -> ThreeMatrix {
    let delta_x = delta_x.normalize();
    
    let beta = delta_x.z.acos();
    let alpha = (-delta_x.y * delta_x.z).atan2(delta_x.x);
    let gamma = delta_x.y.atan2(delta_x.x);

    ThreeMatrix {
        rows: [
            ThreeVector::from_spherical_angles(FRAC_PI_2 - beta, -alpha) * gamma.cos()
            - ThreeVector::from_spherical_angles(FRAC_PI_2, FRAC_PI_2 - alpha) * gamma.sin(),
            ThreeVector::from_spherical_angles(FRAC_PI_2 - beta, -alpha) * gamma.sin()
            + ThreeVector::from_spherical_angles(FRAC_PI_2, FRAC_PI_2 - alpha) * gamma.cos(),
            ThreeVector::from_spherical_angles(beta, PI - alpha),
        ]
    }
}

pub fn rotate_spherical_angles(zenith: f64, azimuth: f64, rotation: ThreeMatrix) -> (f64, f64) {
    let vector = ThreeVector::from_spherical_angles(zenith, azimuth);

    let vector = rotation.dot(&vector);
    let theta = vector.z.acos();
    let phi = vector.y.atan2(vector.x);
    (theta, phi)
}

#[allow(dead_code)]
#[pyfunction]
pub fn rotation_matrix_from_vertices(vertex_1: [f64; 3], vertex_2: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = _rotation_matrix_from_vertices(
        ThreeVector::from_array(&vertex_1),
        ThreeVector::from_array(&vertex_2),
    );
    rotation.to_pyarray()
}

#[allow(dead_code)]
#[pyfunction]
pub fn rotation_matrix_from_delta_x(delta_x: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = _rotation_matrix_from_delta_x(ThreeVector::from_array(&delta_x));
    rotation.to_pyarray()
}

#[allow(dead_code)]
#[pyfunction]
pub fn zenith_azimuth_to_theta_phi(zenith: f64, azimuth: f64, delta_x: [f64; 3]) -> (f64, f64) {
    let rotation = _rotation_matrix_from_delta_x(ThreeVector::from_array(&delta_x));
    rotate_spherical_angles(zenith, azimuth, rotation)
}

#[allow(dead_code)]
#[pyfunction]
pub fn zenith_azimuth_to_theta_phi_optimized(
    zenith: f64, azimuth: f64, vertex_1:[f64; 3], vertex_2: [f64; 3]
) -> (f64, f64) {
    let rotation = _rotation_matrix_from_vertices(
        ThreeVector::from_array(&vertex_1),
        ThreeVector::from_array(&vertex_2),
    );
    rotate_spherical_angles(zenith, azimuth, rotation)
}
