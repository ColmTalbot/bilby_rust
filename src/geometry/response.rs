use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArray2, PyArray3};
use pyo3::{pyfunction, Py, Python};

use super::{
    antenna::DetectorGeometry,
    polarization::polarization_tensor,
    three::{ThreeMatrix, ThreeVector},
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

    let mut output: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
    for (&frequency, &gps_time) in frequencies.iter().zip(gps_times.iter()) {
        output.push(
            det.finite_size_tensor(frequency, gps_time, ra, dec)
                .to_vec(),
        );
    }
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
) -> Py<PyArray1<Complex<f64>>> {
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    let mut output: Vec<Complex<f64>> = Vec::new();

    for (&frequency, &gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pol: ThreeMatrix = polarization_tensor(ra, dec, gps_time, psi, mode);

        let temp: Complex64 = (det.finite_size_tensor(frequency, gps_time, ra, dec) * pol).sum();

        output.push(temp);
    }
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
) -> Py<PyArray2<Complex<f64>>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();
    let det: DetectorGeometry = DetectorGeometry::new(x, y, free_spectral_range);

    let mut output: Vec<Vec<Complex<f64>>> = Vec::new();

    for (&frequency, &gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pols: [ThreeMatrix; 2] = [
            polarization_tensor(ra, dec, gps_time, psi, "plus"),
            polarization_tensor(ra, dec, gps_time, psi, "cross"),
        ];

        let det_tensor = det.finite_size_tensor(frequency, gps_time, ra, dec);

        let mut temp: [Complex<f64>; 2] = [Complex::new(0.0, 0.0); 2];
        for i in 0..2 {
            temp[i] = (det_tensor * pols[i]).sum();
        }
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| PyArray2::from_vec2_bound(py, &output).unwrap().unbind())
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
) -> Py<PyArray2<Complex<f64>>> {
    let x: ThreeVector = x.into();
    let y: ThreeVector = y.into();
    let det: DetectorGeometry = DetectorGeometry::new(x, y, free_spectral_range);

    let mut output: Vec<Vec<Complex<f64>>> = Vec::new();

    for (&frequency, &gps_time) in frequency.iter().zip(gps_time.iter()) {
        let pols: [ThreeMatrix; 6] = [
            polarization_tensor(ra, dec, gps_time, psi, "plus"),
            polarization_tensor(ra, dec, gps_time, psi, "cross"),
            polarization_tensor(ra, dec, gps_time, psi, "breathing"),
            polarization_tensor(ra, dec, gps_time, psi, "longitudinal"),
            polarization_tensor(ra, dec, gps_time, psi, "x"),
            polarization_tensor(ra, dec, gps_time, psi, "y"),
        ];

        let det_tensor = det.finite_size_tensor(frequency, gps_time, ra, dec);

        let mut temp: [Complex<f64>; 6] = [Complex::new(0.0, 0.0); 6];
        for i in 0..6 {
            temp[i] = (det_tensor * pols[i]).sum();
        }
        output.push(temp.to_vec());
    }
    Python::with_gil(|py| PyArray2::from_vec2_bound(py, &output).unwrap().unbind())
}
