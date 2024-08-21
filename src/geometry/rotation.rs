use std::f64::consts::{FRAC_PI_2, PI};

use numpy::PyArray2;
use pyo3::{pyfunction, Py};

use super::three::{SphericalAngles, ThreeMatrix, ThreeVector};

pub fn rotation_matrix_from_vertices(vertex_1: ThreeVector, vertex_2: ThreeVector) -> ThreeMatrix {
    let delta_x = (vertex_1 - vertex_2).normalize();
    let midpoint = (vertex_1 + vertex_2).normalize();
    let x_axis = delta_x.cross(midpoint).normalize();
    let y_axis = x_axis.cross(delta_x).normalize();
    ThreeMatrix {
        rows: [x_axis, y_axis, delta_x],
    }
}

pub fn rotation_matrix_from_delta_x(delta_x: ThreeVector) -> ThreeMatrix {
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
        rows: [angles1, angles2, angles3],
    }
    .rotate_z(gamma)
}

pub fn rotate_spherical_angles(zenith: f64, azimuth: f64, rotation: ThreeMatrix) -> (f64, f64) {
    let vector: ThreeVector = SphericalAngles { zenith, azimuth }.into();

    let vector = rotation.dot(vector);
    let theta = vector.z.acos();
    let phi = vector.y.atan2(vector.x);
    (theta, phi)
}

#[pyfunction(name = "rotation_matrix_from_vertices")]
pub fn _py_rotation_matrix_from_vertices(vertex_1: [f64; 3], vertex_2: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = rotation_matrix_from_vertices(vertex_1.into(), vertex_2.into());
    rotation.into()
}

#[pyfunction(name = "rotation_matrix_from_delta_x")]
pub fn _py_rotation_matrix_from_delta_x(delta_x: [f64; 3]) -> Py<PyArray2<f64>> {
    let rotation = rotation_matrix_from_delta_x(delta_x.into());
    rotation.into()
}

#[pyfunction]
pub fn zenith_azimuth_to_theta_phi(zenith: f64, azimuth: f64, delta_x: [f64; 3]) -> (f64, f64) {
    let rotation = rotation_matrix_from_delta_x(delta_x.into());
    rotate_spherical_angles(zenith, azimuth, rotation)
}

#[pyfunction]
pub fn zenith_azimuth_to_theta_phi_optimized(
    zenith: f64,
    azimuth: f64,
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
) -> (f64, f64) {
    let rotation = rotation_matrix_from_vertices(vertex_1.into(), vertex_2.into());
    rotate_spherical_angles(zenith, azimuth, rotation)
}

#[pyfunction]
pub fn theta_phi_to_zenith_azimuth(theta: f64, phi: f64, delta_x: [f64; 3]) -> (f64, f64) {
    let rotation = rotation_matrix_from_delta_x(delta_x.into());
    rotate_spherical_angles(theta, phi, rotation.transpose())
}

#[pyfunction]
pub fn theta_phi_to_zenith_azimuth_optimized(
    theta: f64,
    phi: f64,
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
) -> (f64, f64) {
    let rotation = rotation_matrix_from_vertices(vertex_1.into(), vertex_2.into());
    rotate_spherical_angles(theta, phi, rotation.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            const EPSILON: f64 = 1e-10;
            assert!(
                ($a.0 - $b.0).abs() < EPSILON,
                "Values not approximately equal: {} vs {}",
                $a.0,
                $b.0
            );
            assert!(
                ($a.1 - $b.1).abs() < EPSILON,
                "Values not approximately equal: {} vs {}",
                $a.1,
                $b.1
            );
        };
    }

    #[test]
    fn test_theta_phi_zenith_azimuth_inverse_delta() {
        let (theta, phi): (f64, f64) = (1.0, 2.0);
        let delta: [f64; 3] = [0.1, 0.5, 3.0];
        let (zenith, azimuth) = theta_phi_to_zenith_azimuth(theta, phi, delta);
        let result: (f64, f64) = zenith_azimuth_to_theta_phi(zenith, azimuth, delta);
        assert_approx_eq!(result, (theta, phi));
    }

    #[test]
    fn test_theta_phi_zenith_azimuth_inverse_vertices() {
        let (theta, phi): (f64, f64) = (1.0, 2.0);
        let vertex_1: [f64; 3] = [0.1, 0.5, 3.0];
        let vertex_2: [f64; 3] = [0.4, 0.1, 1.0];
        let (zenith, azimuth) =
            theta_phi_to_zenith_azimuth_optimized(theta, phi, vertex_1, vertex_2);
        let result: (f64, f64) =
            zenith_azimuth_to_theta_phi_optimized(zenith, azimuth, vertex_1, vertex_2);
        assert_approx_eq!(result, (theta, phi));
    }
}
