/// # bilby_rust
///
/// `bilby_rust` provides implementations of gravitational-wave detector geometry.
///
/// This is primarily intended to be used as a dependency for the `bilby` gravitational-wave
/// parameter estimation package. However, it can also be used as a standalone package for
/// computing detector responses and antenna patterns.
use pyo3::types::PyModule;
use pyo3::{py_run, pymodule, wrap_pyfunction, Bound, PyResult};

pub mod geometry;
pub mod time;

use geometry::antenna::{
    calculate_arm, detector_tensor, get_polarization_tensor, time_delay_from_geocenter,
    time_delay_from_geocenter_vectorized, time_delay_geocentric,
    time_dependent_polarization_tensor,
};
use geometry::response::{
    antenna_response, antenna_response_all_modes, antenna_response_multiple_modes,
    antenna_response_tensor_modes, frequency_dependent_detector_tensor,
};
use geometry::rotation::{
    _py_rotation_matrix_from_delta_x, _py_rotation_matrix_from_vertices, theta_phi_to_zenith_azimuth,
    theta_phi_to_zenith_azimuth_optimized, zenith_azimuth_to_theta_phi,
    zenith_azimuth_to_theta_phi_optimized,
};
use geometry::util::ra_dec_to_theta_phi;
use time::{
    greenwich_mean_sidereal_time, greenwich_mean_sidereal_time_vectorized, greenwich_sidereal_time,
    n_leap_seconds, utc_to_julian_day,
};

#[pymodule]
fn bilby_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__all__", ["time", "geometry"])?;

    let t_: Bound<'_, PyModule> = PyModule::new_bound(m.py(), "time")?;
    // see https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
    py_run!(
        m.py(),
        t_,
        "import sys; sys.modules['bilby_rust.time'] = t_"
    );
    t_.add_function(wrap_pyfunction!(time::gps_time_to_utc, &t_)?)?;
    t_.add_function(wrap_pyfunction!(greenwich_mean_sidereal_time, &t_)?)?;
    t_.add_function(wrap_pyfunction!(
        greenwich_mean_sidereal_time_vectorized,
        &t_
    )?)?;
    t_.add_function(wrap_pyfunction!(greenwich_sidereal_time, &t_)?)?;
    t_.add_function(wrap_pyfunction!(n_leap_seconds, &t_)?)?;
    t_.add_function(wrap_pyfunction!(utc_to_julian_day, &t_)?)?;
    m.add_submodule(&t_)?;

    let g_: Bound<'_, PyModule> = PyModule::new_bound(m.py(), "geometry")?;
    py_run!(
        m.py(),
        g_,
        "import sys; sys.modules['bilby_rust.geometry'] = g_"
    );
    g_.add_function(wrap_pyfunction!(antenna_response, &g_)?)?;
    g_.add_function(wrap_pyfunction!(antenna_response_all_modes, &g_)?)?;
    g_.add_function(wrap_pyfunction!(antenna_response_multiple_modes, &g_)?)?;
    g_.add_function(wrap_pyfunction!(antenna_response_tensor_modes, &g_)?)?;
    g_.add_function(wrap_pyfunction!(calculate_arm, &g_)?)?;
    g_.add_function(wrap_pyfunction!(detector_tensor, &g_)?)?;
    g_.add_function(wrap_pyfunction!(frequency_dependent_detector_tensor, &g_)?)?;
    g_.add_function(wrap_pyfunction!(get_polarization_tensor, &g_)?)?;
    g_.add_function(wrap_pyfunction!(ra_dec_to_theta_phi, &g_)?)?;
    g_.add_function(wrap_pyfunction!(_py_rotation_matrix_from_delta_x, &g_)?)?;
    g_.add_function(wrap_pyfunction!(_py_rotation_matrix_from_vertices, &g_)?)?;
    g_.add_function(wrap_pyfunction!(time_delay_geocentric, &g_)?)?;
    g_.add_function(wrap_pyfunction!(time_delay_from_geocenter, &g_)?)?;
    g_.add_function(wrap_pyfunction!(time_delay_from_geocenter_vectorized, &g_)?)?;
    g_.add_function(wrap_pyfunction!(time_dependent_polarization_tensor, &g_)?)?;
    g_.add_function(wrap_pyfunction!(theta_phi_to_zenith_azimuth, &g_)?)?;
    g_.add_function(wrap_pyfunction!(
        theta_phi_to_zenith_azimuth_optimized,
        &g_
    )?)?;
    g_.add_function(wrap_pyfunction!(zenith_azimuth_to_theta_phi, &g_)?)?;
    g_.add_function(wrap_pyfunction!(
        zenith_azimuth_to_theta_phi_optimized,
        &g_
    )?)?;
    m.add_submodule(&g_)?;

    Ok(())
}
