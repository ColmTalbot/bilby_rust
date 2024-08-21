use std::f64::consts::PI;

use pyo3::pyfunction;

use super::{
    three::{SphericalAngles, ThreeVector},
    time,
};

/// Convert from right ascension and declination to spherical angles (theta, phi)
/// that are fixed wrt the Earth.
/// 
/// The angles are defined as:
/// 
/// - `theta` is the zenith angle (pi / 2 is the equator)
/// - `phi` is the azimuthal angle (0 is the Greenwich meridian)
#[pyfunction]
pub fn ra_dec_to_theta_phi(ra: f64, dec: f64, gps_time: f64) -> (f64, f64) {
    let gmst = time::greenwich_mean_sidereal_time(gps_time) % (2.0 * PI);
    let theta = PI / 2.0 - dec;
    let phi = ra - gmst;
    (theta, phi)
}

/// Convert from spherical angles (theta, phi) to right ascension and declination
/// 
/// The angles are defined as:
/// 
/// - `theta` is the zenith angle (pi / 2 is the equator)
/// - `phi` is the azimuthal angle (0 is the Greenwich meridian)
#[pyfunction]
pub fn theta_phi_to_ra_dec(theta: f64, phi: f64, gps_time: f64) -> (f64, f64) {
    let gmst = time::greenwich_mean_sidereal_time(gps_time) % (2.0 * PI);
    let ra = phi + gmst;
    let dec = PI / 2.0 - theta;
    (ra, dec)
}

/// Compute the line of sight vector from right ascension and declination.
/// 
/// This is sometimes denoted `\hat{N}` or `\hat{n}`.
pub fn line_of_sight(ra: f64, dec: f64, gps_time: f64) -> ThreeVector {
    let theta_phi: SphericalAngles = ra_dec_to_theta_phi(ra, dec, gps_time).into();
    theta_phi.into()
}
