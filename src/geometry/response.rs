/// Antenna response functions including time- and frequency-dependent
/// responses in addition to the standard time- and frequency-independent responses.
use num_complex::Complex;

use super::{
    antenna::DetectorGeometry,
    polarization::{PolarizationMatrix, ALL_MODES, TENSOR_MODES},
    three::ComplexThreeMatrix,
    util::ra_dec_to_theta_phi,
};

pub fn frequency_dependent_detector_tensor(
    x: [f64; 3],
    y: [f64; 3],
    frequencies: &[f64],
    ra: f64,
    dec: f64,
    gps_times: &[f64],
    free_spectral_range: f64,
) -> Vec<ComplexThreeMatrix> {
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    frequencies
        .iter()
        .zip(gps_times.iter())
        .map(|(&frequency, &gps_time)| det.finite_size_tensor(frequency, gps_time, ra, dec))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn antenna_response(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: &[f64],
    psi: f64,
    mode: &str,
    frequency: &[f64],
    free_spectral_range: f64,
) -> Vec<Complex<f64>> {
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    frequency
        .iter()
        .zip(gps_time.iter())
        .map(|(&frequency, &gps_time)| {
            let (theta, phi) = ra_dec_to_theta_phi(ra, dec, gps_time);
            let pol: PolarizationMatrix = PolarizationMatrix::new(theta, phi, psi);
            (det.finite_size_tensor(frequency, gps_time, ra, dec) * pol.mode(mode)).sum()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn antenna_response_tensor_modes(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: &[f64],
    psi: f64,
    frequency: &[f64],
    free_spectral_range: f64,
) -> Vec<Vec<Complex<f64>>> {
    antenna_response_multiple_modes(
        x,
        y,
        ra,
        dec,
        gps_time,
        psi,
        frequency,
        free_spectral_range,
        &TENSOR_MODES,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn antenna_response_all_modes(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: &[f64],
    psi: f64,
    frequency: &[f64],
    free_spectral_range: f64,
) -> Vec<Vec<Complex<f64>>> {
    antenna_response_multiple_modes(
        x,
        y,
        ra,
        dec,
        gps_time,
        psi,
        frequency,
        free_spectral_range,
        &ALL_MODES,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn antenna_response_multiple_modes<S: AsRef<str>>(
    x: [f64; 3],
    y: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: &[f64],
    psi: f64,
    frequency: &[f64],
    free_spectral_range: f64,
    modes: &[S],
) -> Vec<Vec<Complex<f64>>> {
    let det: DetectorGeometry = DetectorGeometry::new(x.into(), y.into(), free_spectral_range);

    frequency
        .iter()
        .zip(gps_time.iter())
        .map(|(&frequency, &gps_time)| {
            let (theta, phi) = ra_dec_to_theta_phi(ra, dec, gps_time);
            let pol: PolarizationMatrix = PolarizationMatrix::new(theta, phi, psi);

            let det_tensor = det.finite_size_tensor(frequency, gps_time, ra, dec);
            modes
                .iter()
                .map(|mode| (det_tensor * pol.mode(mode.as_ref())).sum())
                .collect()
        })
        .collect()
}
