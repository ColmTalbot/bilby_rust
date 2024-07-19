use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArray2, PyArray3};
use pyo3::{pyfunction, Py, Python};

use super::{
    antenna::DetectorGeometry,
    polarization::{PolarizationMatrix, ALL_MODES, TENSOR_MODES},
    util::ra_dec_to_theta_phi,
};

#[allow(dead_code)]
#[pyfunction]
pub fn frequency_dependent_detector_tensor(
    x: [f64; 3],
    y: [f64; 3],
    frequencies: Vec<f64>,
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray3<Complex64>> {
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    let output: Vec<Vec<Vec<Complex<f64>>>> = frequencies
        .iter()
        .zip(gps_times.iter())
        .map(|(&frequency, &gps_time)| {
            det.finite_size_tensor(frequency, gps_time, ra, dec)
                .to_vec()
        })
        .collect();
    Python::with_gil(|py| PyArray3::from_vec3_bound(py, &output).unwrap().unbind())
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response(
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
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    let output: Vec<Complex<f64>> = frequency
        .iter()
        .zip(gps_time.iter())
        .map(|(&frequency, &gps_time)| {
            let (theta, phi) = ra_dec_to_theta_phi(ra, dec, gps_time);
            let pol: PolarizationMatrix = PolarizationMatrix::new(theta, phi, psi);
            (det.finite_size_tensor(frequency, gps_time, ra, dec) * pol.mode(mode)).sum()
        })
        .collect();
    Python::with_gil(|py| PyArray1::from_vec_bound(py, output).unbind())
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response_tensor_modes(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray2<Complex64>> {
    antenna_response_multiple_modes(
        x,
        y,
        ra,
        dec,
        gps_time,
        psi,
        frequency,
        free_spectral_range,
        TENSOR_MODES.iter().map(|&s| s.to_string()).collect(),
    )
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response_all_modes(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: Vec<f64>,
    psi: f64,
    frequency: Vec<f64>,
    free_spectral_range: f64,
) -> Py<PyArray2<Complex64>> {
    antenna_response_multiple_modes(
        x,
        y,
        ra,
        dec,
        gps_time,
        psi,
        frequency,
        free_spectral_range,
        ALL_MODES.iter().map(|&s| s.to_string()).collect(),
    )
}

#[allow(clippy::too_many_arguments, dead_code)]
#[pyfunction]
pub fn antenna_response_multiple_modes(
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
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    let output: Vec<Vec<Complex<f64>>> = frequency
        .iter()
        .zip(gps_time.iter())
        .map(|(&frequency, &gps_time)| {
            let (theta, phi) = ra_dec_to_theta_phi(ra, dec, gps_time);
            let pol: PolarizationMatrix = PolarizationMatrix::new(theta, phi, psi);

            let det_tensor = det.finite_size_tensor(frequency, gps_time, ra, dec);
            modes
                .iter()
                .map(|mode| (det_tensor * pol.mode(mode)).sum()
            )
                .collect()
        })
        .collect();
    Python::with_gil(|py| PyArray2::from_vec2_bound(py, &output).unwrap().unbind())
}
