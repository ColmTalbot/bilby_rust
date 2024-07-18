use std::f64::consts::{FRAC_PI_2, PI};

use numpy::PyArray2;
use pyo3::{pyfunction, Py};

use super::util::{SphericalAngles, ThreeMatrix, ThreeVector};

pub fn _rotation_matrix_from_vertices(vertex_1: ThreeVector, vertex_2: ThreeVector) -> ThreeMatrix {
    let delta_x = (vertex_1 - vertex_2).normalize();
    let midpoint = (vertex_1 + vertex_2).normalize();
    let x_axis = delta_x.cross(midpoint);
    let y_axis = x_axis.cross(delta_x);
    ThreeMatrix {
        rows: [x_axis, y_axis, delta_x],
    }
}

pub fn _rotation_matrix_from_delta_x(delta_x: ThreeVector) -> ThreeMatrix {
    let delta_x = delta_x.normalize();

    let beta = delta_x.z.acos();
    let alpha = (-delta_x.y * delta_x.z).atan2(delta_x.x);
    let gamma = delta_x.y.atan2(delta_x.x);

    let angles1: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2 - beta,
        azimuth: -alpha,
    }
    .into();
    let angles2: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2,
        azimuth: FRAC_PI_2 - alpha,
    }
    .into();
    let angles3: ThreeVector = SphericalAngles {
        zenith: beta,
        azimuth: PI - alpha,
    }
    .into();

    ThreeMatrix {
        rows: [
            angles1 * gamma.cos() - angles2 * gamma.sin(),
            angles1 * gamma.sin() + angles2 * gamma.cos(),
            angles3,
        ],
    }
}

pub fn rotate_spherical_angles(zenith: f64, azimuth: f64, rotation: ThreeMatrix) -> (f64, f64) {
    let vector: ThreeVector = SphericalAngles { zenith, azimuth }.into();

    let vector = rotation.dot(vector);
    let theta = vector.z.acos();
    let phi = vector.y.atan2(vector.x);
    (theta, phi)
}

#[allow(dead_code)]
#[pyfunction]
pub fn rotation_matrix_from_vertices(vertex_1: [f64; 3], vertex_2: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = _rotation_matrix_from_vertices(vertex_1.into(), vertex_2.into());
    rotation.into()
}

#[allow(dead_code)]
#[pyfunction]
pub fn rotation_matrix_from_delta_x(delta_x: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = _rotation_matrix_from_delta_x(delta_x.into());
    rotation.into()
}

#[allow(dead_code)]
#[pyfunction]
pub fn zenith_azimuth_to_theta_phi(zenith: f64, azimuth: f64, delta_x: [f64; 3]) -> (f64, f64) {
    let rotation = _rotation_matrix_from_delta_x(delta_x.into());
    rotate_spherical_angles(zenith, azimuth, rotation)
}

#[allow(dead_code)]
#[pyfunction]
pub fn zenith_azimuth_to_theta_phi_optimized(
    zenith: f64,
    azimuth: f64,
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
) -> (f64, f64) {
    let rotation = _rotation_matrix_from_vertices(vertex_1.into(), vertex_2.into());
    rotate_spherical_angles(zenith, azimuth, rotation)
}
