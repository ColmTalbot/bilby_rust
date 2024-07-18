use std::f64::consts::PI;

use pyo3::pyfunction;

use super::time;
mod antenna;
mod util;
mod polarization;
mod response;
mod rotation;

#[allow(dead_code)]
#[pyfunction]
pub fn ra_dec_to_theta_phi(ra: f64, dec: f64, gps_time: f64) -> (f64, f64) {
    let gmst = time::greenwich_mean_sidereal_time(gps_time) % (2.0 * PI);
    let theta = PI / 2.0 - dec;
    let phi = ra - gmst;
    (theta, phi)
}

#[allow(unused_imports)]
pub use crate::geometry::antenna::{
    calculate_arm,
    detector_tensor,
    get_polarization_tensor,
    time_delay_from_geocenter,
    time_delay_from_geocenter_vectorized,
    time_delay_geocentric,
    time_dependent_polarization_tensor,
};
#[allow(unused_imports)]
pub use crate::geometry::response::{
    antenna_response,
    antenna_response_all_modes,
    antenna_response_tensor_modes,
    frequency_dependent_detector_tensor,
};
#[allow(unused_imports)]
pub use crate::geometry::rotation::{
    rotation_matrix_from_delta_x,
    rotation_matrix_from_vertices,
    zenith_azimuth_to_theta_phi,
    zenith_azimuth_to_theta_phi_optimized,
};
