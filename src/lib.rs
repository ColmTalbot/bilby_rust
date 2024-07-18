use pyo3::{Bound, PyResult, pymodule, py_run, wrap_pyfunction};
use pyo3::types::PyModule;

mod time;
mod geometry;

#[pymodule]
fn bilby_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__all__", ["time", "geometry"])?;

    let time_: Bound<PyModule> = PyModule::new_bound(m.py(), "time")?;
    // see https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
    py_run!(m.py(), time_, "import sys; sys.modules['bilby_rust.time'] = time_");
    time_.add_function(wrap_pyfunction!(time::gps_time_to_utc, &time_)?)?;
    time_.add_function(wrap_pyfunction!(time::greenwich_mean_sidereal_time, &time_)?)?;
    time_.add_function(wrap_pyfunction!(time::greenwich_mean_sidereal_time_vectorized, &time_)?)?;
    time_.add_function(wrap_pyfunction!(time::greenwich_sidereal_time, &time_)?)?;
    time_.add_function(wrap_pyfunction!(time::n_leap_seconds, &time_)?)?;
    time_.add_function(wrap_pyfunction!(time::utc_to_julian_day, &time_)?)?;
    m.add_submodule(&time_)?;

    let geometry_: Bound<PyModule> = PyModule::new_bound(m.py(), "geometry")?;
    py_run!(m.py(), geometry_, "import sys; sys.modules['bilby_rust.geometry'] = geometry_");
    geometry_.add_function(wrap_pyfunction!(geometry::antenna_response, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::antenna_response_all_modes, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::antenna_response_tensor_modes, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::calculate_arm, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::detector_tensor, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::frequency_dependent_detector_tensor, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::get_polarization_tensor, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::ra_dec_to_theta_phi, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::rotation_matrix_from_delta_x, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::rotation_matrix_from_vertices, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::time_delay_geocentric, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::time_delay_from_geocenter, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::time_delay_from_geocenter_vectorized, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::time_dependent_polarization_tensor, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::zenith_azimuth_to_theta_phi, &geometry_)?)?;
    geometry_.add_function(wrap_pyfunction!(geometry::zenith_azimuth_to_theta_phi_optimized, &geometry_)?)?;
    m.add_submodule(&geometry_)?;

    Ok(())
}
