use std::f64::consts::PI;

use pyo3::pyfunction;

use super::{
    three::{SphericalAngles, ThreeVector},
    time,
};

#[allow(dead_code)]
#[pyfunction]
pub fn ra_dec_to_theta_phi(ra: f64, dec: f64, gps_time: f64) -> (f64, f64) {
    let gmst = time::greenwich_mean_sidereal_time(gps_time) % (2.0 * PI);
    let theta = PI / 2.0 - dec;
    let phi = ra - gmst;
    (theta, phi)
}

pub fn line_of_sight(ra: f64, dec: f64, gps_time: f64) -> ThreeVector {
    let theta_phi: SphericalAngles = ra_dec_to_theta_phi(ra, dec, gps_time).into();
    theta_phi.into()
}
