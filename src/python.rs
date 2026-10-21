//! `Python` bindings, enabled with the `python` feature.

#![allow(clippy::too_many_arguments)]

use chrono::{DateTime, Utc};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{py_run, pyfunction, pymodule, wrap_pyfunction, Bound, Py, PyErr, PyResult, Python};

use crate::geometry::{
    antenna, polarization,
    response,
    rotation,
    three::{ComplexThreeMatrix, ThreeMatrix, ThreeVector},
    util,
};
use crate::time;

fn vector_to_py(py: Python<'_>, vector: ThreeVector) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, vec![vector.x, vector.y, vector.z]).unbind()
}

fn matrix_to_py(py: Python<'_>, matrix: ThreeMatrix) -> Py<PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = matrix.into();
    PyArray2::from_vec2(py, &rows).unwrap().unbind()
}

fn complex_tensors_to_py(
    py: Python<'_>,
    tensors: Vec<ComplexThreeMatrix>,
) -> Py<PyArray3<Complex64>> {
    let output: Vec<Vec<Vec<Complex64>>> = tensors.into_iter().map(|t| t.to_vec()).collect();
    PyArray3::from_vec3(py, &output).unwrap().unbind()
}

fn complex_rows_to_py(py: Python<'_>, rows: Vec<Vec<Complex64>>) -> Py<PyArray2<Complex64>> {
    PyArray2::from_vec2(py, &rows).unwrap().unbind()
}

#[pyfunction]
fn n_leap_seconds(s: i32) -> i32 {
    time::n_leap_seconds(s)
}

#[pyfunction]
fn gps_time_to_utc(gps_time: i32) -> DateTime<Utc> {
    time::gps_time_to_utc(gps_time)
}

#[pyfunction]
fn utc_to_julian_day(time: DateTime<Utc>) -> f64 {
    crate::time::utc_to_julian_day(time)
}

#[pyfunction]
fn greenwich_sidereal_time(gps_time: f64, equation_of_equinoxes: f64) -> f64 {
    time::greenwich_sidereal_time(gps_time, equation_of_equinoxes)
}

#[pyfunction]
fn greenwich_mean_sidereal_time(gps_time: f64) -> f64 {
    time::greenwich_mean_sidereal_time(gps_time)
}

#[pyfunction]
fn greenwich_mean_sidereal_time_vectorized(py: Python<'_>, gps_times: Vec<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, time::greenwich_mean_sidereal_time_vectorized(&gps_times)).unbind()
}

#[pyfunction]
fn get_polarization_tensor(
    py: Python<'_>,
    ra: f64,
    dec: f64,
    gps_time: f64,
    psi: f64,
    mode: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    std::panic::catch_unwind(|| polarization::polarization_tensor(ra, dec, gps_time, psi, mode))
        .map(|tensor| matrix_to_py(py, tensor))
        .map_err(|_| {
            PyErr::new::<PyValueError, _>(
                "Error in get_polarization_tensor: {mode} not a polarization mode!",
            )
        })
}

#[pyfunction]
fn time_delay_geocentric(
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: f64,
) -> f64 {
    antenna::time_delay_geocentric(vertex_1, vertex_2, ra, dec, gps_time)
}

#[pyfunction]
fn time_delay_from_geocenter(vertex: [f64; 3], ra: f64, dec: f64, gps_time: f64) -> f64 {
    antenna::time_delay_from_geocenter(vertex, ra, dec, gps_time)
}

#[pyfunction]
fn time_delay_from_geocenter_vectorized(
    py: Python<'_>,
    vertex: [f64; 3],
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
) -> Py<PyArray1<f64>> {
    let delays = antenna::time_delay_from_geocenter_vectorized(vertex, ra, dec, &gps_times);
    PyArray1::from_vec(py, delays).unbind()
}

#[pyfunction]
fn calculate_arm(
    py: Python<'_>,
    arm_tilt: f64,
    arm_azimuth: f64,
    longitude: f64,
    latitude: f64,
) -> Py<PyArray1<f64>> {
    vector_to_py(
        py,
        antenna::calculate_arm(arm_tilt, arm_azimuth, longitude, latitude),
    )
}

#[pyfunction]
fn detector_tensor(py: Python<'_>, x: [f64; 3], y: [f64; 3]) -> Py<PyArray2<f64>> {
    matrix_to_py(py, antenna::detector_tensor(x, y))
}

#[pyfunction]
fn time_dependent_polarization_tensor(
    py: Python<'_>,
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
    psi: f64,
    mode: &str,
) -> Py<PyArray3<f64>> {
    let output: Vec<Vec<Vec<f64>>> =
        antenna::time_dependent_polarization_tensor(ra, dec, &gps_times, psi, mode)
            .into_iter()
            .map(Into::into)
            .collect();
    PyArray3::from_vec3(py, &output).unwrap().unbind()
}

#[pyfunction]
fn frequency_dependent_detector_tensor(
    py: Python<'_>,
    x: [f64; 3],
    y: [f64; 3],
    frequencies: Vec<f64>,
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray3<Complex64>> {
    complex_tensors_to_py(
        py,
        response::frequency_dependent_detector_tensor(
            x,
            y,
            &frequencies,
            ra,
            dec,
            &gps_times,
            free_spectral_range,
        ),
    )
}

#[pyfunction]
fn antenna_response(
    py: Python<'_>,
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    mode: &str,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray1<Complex64>> {
    let output = response::antenna_response(
        x,
        y,
        ra,
        dec,
        &gps_time,
        psi,
        mode,
        &frequency,
        free_spectral_range,
    );
    PyArray1::from_vec(py, output).unbind()
}

#[pyfunction]
fn antenna_response_tensor_modes(
    py: Python<'_>,
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray2<Complex64>> {
    complex_rows_to_py(
        py,
        response::antenna_response_tensor_modes(
            x,
            y,
            ra,
            dec,
            &gps_time,
            psi,
            &frequency,
            free_spectral_range,
        ),
    )
}

#[pyfunction]
fn antenna_response_all_modes(
    py: Python<'_>,
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray2<Complex64>> {
    complex_rows_to_py(
        py,
        response::antenna_response_all_modes(
            x,
            y,
            ra,
            dec,
            &gps_time,
            psi,
            &frequency,
            free_spectral_range,
        ),
    )
}

#[pyfunction]
fn antenna_response_multiple_modes(
    py: Python<'_>,
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
    modes: Vec<String>,
) -> Py<PyArray2<Complex64>> {
    complex_rows_to_py(
        py,
        response::antenna_response_multiple_modes(
            x,
            y,
            ra,
            dec,
            &gps_time,
            psi,
            &frequency,
            free_spectral_range,
            &modes,
        ),
    )
}

#[pyfunction]
fn ra_dec_to_theta_phi(ra: f64, dec: f64, gps_time: f64) -> (f64, f64) {
    util::ra_dec_to_theta_phi(ra, dec, gps_time)
}

#[pyfunction]
fn rotation_matrix_from_vertices(
    py: Python<'_>,
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
) -> Py<PyArray2<f64>> {
    matrix_to_py(
        py,
        rotation::rotation_matrix_from_vertices(vertex_1.into(), vertex_2.into()),
    )
}

#[pyfunction]
fn rotation_matrix_from_delta_x(py: Python<'_>, delta_x: [f64; 3]) -> Py<PyArray2<f64>> {
    matrix_to_py(py, rotation::rotation_matrix_from_delta_x(delta_x.into()))
}

#[pyfunction]
fn zenith_azimuth_to_theta_phi(zenith: f64, azimuth: f64, delta_x: [f64; 3]) -> (f64, f64) {
    rotation::zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
}

#[pyfunction]
fn zenith_azimuth_to_theta_phi_optimized(
    zenith: f64,
    azimuth: f64,
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
) -> (f64, f64) {
    rotation::zenith_azimuth_to_theta_phi_optimized(zenith, azimuth, vertex_1, vertex_2)
}

#[pyfunction]
fn theta_phi_to_zenith_azimuth(theta: f64, phi: f64, delta_x: [f64; 3]) -> (f64, f64) {
    rotation::theta_phi_to_zenith_azimuth(theta, phi, delta_x)
}

#[pyfunction]
fn theta_phi_to_zenith_azimuth_optimized(
    theta: f64,
    phi: f64,
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
) -> (f64, f64) {
    rotation::theta_phi_to_zenith_azimuth_optimized(theta, phi, vertex_1, vertex_2)
}

#[pymodule(gil_used = false)]
fn bilby_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__all__", ["time", "geometry"])?;

    let t_: Bound<'_, PyModule> = PyModule::new(m.py(), "time")?;
    // see https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
    py_run!(
        m.py(),
        t_,
        "import sys; sys.modules['bilby_rust.time'] = t_"
    );
    t_.add_function(wrap_pyfunction!(gps_time_to_utc, &t_)?)?;
    t_.add_function(wrap_pyfunction!(greenwich_mean_sidereal_time, &t_)?)?;
    t_.add_function(wrap_pyfunction!(
        greenwich_mean_sidereal_time_vectorized,
        &t_
    )?)?;
    t_.add_function(wrap_pyfunction!(greenwich_sidereal_time, &t_)?)?;
    t_.add_function(wrap_pyfunction!(n_leap_seconds, &t_)?)?;
    t_.add_function(wrap_pyfunction!(utc_to_julian_day, &t_)?)?;
    m.add_submodule(&t_)?;

    let g_: Bound<'_, PyModule> = PyModule::new(m.py(), "geometry")?;
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
    g_.add_function(wrap_pyfunction!(rotation_matrix_from_delta_x, &g_)?)?;
    g_.add_function(wrap_pyfunction!(rotation_matrix_from_vertices, &g_)?)?;
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
