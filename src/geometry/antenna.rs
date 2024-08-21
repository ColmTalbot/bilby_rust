use std::f64::consts::{FRAC_PI_2, PI};

use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{exceptions::PyValueError, pyfunction, Py, PyErr, PyResult, Python};

const SPEED_OF_LIGHT_IN_VACUUM: f64 = 299_792_458.0;

use super::{
    polarization::polarization_tensor,
    three::{ComplexThreeMatrix, SphericalAngles, ThreeMatrix, ThreeVector},
    util::{line_of_sight, ra_dec_to_theta_phi},
};

/// Represents the geometry of a gravitational wave detector.
///
/// The detector geometry is defined by two vectors, `x` and `y`, which are the unit vectors pointing
/// in the direction of the two arms of the detector. The `free_spectral_range` is the frequency
/// corresponding to a phase difference of π traveling down an arm of the detector.
pub struct DetectorGeometry {
    /// The unit vector pointing in the direction of the first arm of the detector.
    x: ThreeVector,
    /// The unit vector pointing in the direction of the second arm of the detector.
    y: ThreeVector,
    /// The frequency corresponding to a phase difference of π traveling down an arm of the detector
    /// $f_{\rm FSR} = \frac{c}{L}$ where $c$ is the speed of light and $L$ is the length of the arms.
    free_spectral_range: f64,
    /// The outer product of the `x` vector with itself. This is used to construct the detector tensor.
    x_tensor: ThreeMatrix,
    /// The outer product of the `y` vector with itself. This is used to construct the detector tensor.
    y_tensor: ThreeMatrix,
    /// The detector tensor neglecting finite-size effects
    /// $D_{ij} = \frac{x_i x_j - y_i y_j}{2}$.
    /// This assumes the x- and y-axes are orthogonal.
    detector_tensor: ThreeMatrix,
}

impl DetectorGeometry {
    /// Construct a new `DetectorGeometry` object from the arms and free spectral range.
    ///
    /// # Arguments
    ///
    /// * `x` - A `ThreeVector` representing the unit vector pointing in the direction of the first arm of the detector.
    /// * `y` - A `ThreeVector` representing the unit vector pointing in the direction of the second arm of the detector.
    /// * `free_spectral_range` - The frequency corresponding to a phase difference of π traveling down an arm of the detector.
    pub fn new(x: ThreeVector, y: ThreeVector, free_spectral_range: f64) -> Self {
        let x_tensor = x.outer(x);
        let y_tensor = y.outer(y);
        Self {
            x,
            y,
            free_spectral_range,
            x_tensor,
            y_tensor,
            detector_tensor: (x_tensor - y_tensor) / 2.0,
        }
    }

    /// Calculates the detector tensor for a given frequency, GPS time, and sky location
    /// including finite-size effects (see [Essick et al. 2017](https://arxiv.org/abs/1708.06843)).
    ///
    /// # Arguments
    ///
    /// * `frequency` - The frequency at which to calculate the tensor.
    /// * `gps_time` - The GPS time used for calculating the line of sight.
    /// * `ra` - The right ascension of the source in radians.
    /// * `dec` - The declination of the source in radians.
    ///
    /// # Returns
    ///
    /// Returns a `ComplexThreeMatrix` representing the finite size tensor.
    pub fn finite_size_tensor(
        &self,
        frequency: f64,
        gps_time: f64,
        ra: f64,
        dec: f64,
    ) -> ComplexThreeMatrix {
        let line_of_sight = line_of_sight(ra, dec, gps_time);
        let cos_xangle = self.x.dot(line_of_sight);
        let cos_yangle = self.y.dot(line_of_sight);
        let delta_x = directional_response(frequency, cos_xangle, self.free_spectral_range);
        let delta_y = directional_response(frequency, cos_yangle, self.free_spectral_range);

        self.x_tensor * delta_x - self.y_tensor * delta_y
    }
}

/// Compute the directional response of a detector arm to a gravitational wave.
///
/// The directional response is given by (see [Essick et al. 2017](https://arxiv.org/abs/1708.06843))
/// $$
/// \delta(\omega, \cos\theta) = \frac{1}{4\omega}
/// \left( \frac{1 - e^{-i\omega(1 - \cos\theta)}}{1 - \cos\theta}
/// - e^{-2i\omega} \frac{1 - e^{i\omega(1 + \cos\theta)}}{1 + \cos\theta} \right)
/// $$
///
/// The limit as $f$ approaches zero is $\delta(0, \cos\theta) = \frac{1}{2}$.
///
/// # Arguments
///
/// * `frequency` - The frequency of the gravitational wave.
/// * `cos_angle` - The cosine of the angle between the detector arm and the line of sight to the source.
/// * `free_spectral_range` - The frequency corresponding to a phase difference of π traveling down an arm of the detector.
///
/// # Returns
///
/// Returns a `Complex<f64>` representing the directional response.
fn directional_response(frequency: f64, cos_angle: f64, free_spectral_range: f64) -> Complex<f64> {
    let omega = Complex::I * PI * frequency / free_spectral_range;
    1.0 / (4.0 * omega)
        * ((1.0 - (-(1.0 - cos_angle) * omega).exp()) / (1.0 - cos_angle)
            - (-2.0 * omega).exp() * (1.0 - ((1.0 + cos_angle) * omega).exp()) / (1.0 + cos_angle))
}

const GEOCENTER: ThreeVector = ThreeVector {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

/// Calculate a polarization tensor for a given sky location and GPS time.
///
/// # Arguments
///
/// * `ra` - The right ascension of the source in radians.
/// * `dec` - The declination of the source in radians.
/// * `gps_time` - The GPS time of the observation.
/// * `psi` - The polarization angle of the source.
/// * `mode` - The polarization mode to calculate. This can be one of `'plus'`, `'cross'`,
/// `'longitudinal'`, `'breathing'`, `'x'`, or `'y'`.
///
/// # Returns
///
/// A `ThreeMatrix` representing the polarization tensor or a 3x3 `numpy` array
/// when called through the `Python` bindings.
#[pyfunction]
pub fn get_polarization_tensor(
    ra: f64,
    dec: f64,
    gps_time: f64,
    psi: f64,
    mode: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let result =
        std::panic::catch_unwind(|| polarization_tensor(ra, dec, gps_time, psi, mode).into());

    match result {
        Ok(array) => Ok(array),
        Err(_) => Err(PyErr::new::<PyValueError, _>(
            "Error in get_polarization_tensor: {mode} not a polarization mode!",
        )),
    }
}

/// Calculate the time delay between two vertices for a given sky location and GPS time.
///
/// # Arguments
///
/// * `vertex_1` - The first vertex as a 3-element array.
/// * `vertex_2` - The second vertex as a 3-element array.
/// * `ra` - The right ascension of the source in radians.
/// * `dec` - The declination of the source in radians.
/// * `gps_time` - The GPS time of the observation.
///
/// # Returns
///
/// The time delay between the two vertices in seconds.
#[pyfunction]
pub fn time_delay_geocentric(
    vertex_1: [f64; 3],
    vertex_2: [f64; 3],
    ra: f64,
    dec: f64,
    gps_time: f64,
) -> f64 {
    _time_delay_from_vertices(vertex_1.into(), vertex_2.into(), ra, dec, gps_time)
}

fn _time_delay_from_vertices(
    vertex_1: ThreeVector,
    vertex_2: ThreeVector,
    ra: f64,
    dec: f64,
    gps_time: f64,
) -> f64 {
    let theta_phi: SphericalAngles = ra_dec_to_theta_phi(ra, dec, gps_time).into();

    (vertex_2 - vertex_1).dot(theta_phi.into()) / SPEED_OF_LIGHT_IN_VACUUM
}

/// Calculate the time delay between a vertex and the geocenter for a given sky location and GPS time.
///
/// # Arguments
///
/// * `vertex` - The vertex as a 3-element array.
/// * `ra` - The right ascension of the source in radians.
/// * `dec` - The declination of the source in radians.
/// * `gps_time` - The GPS time of the observation.
///
/// # Returns
///
/// The time delay between the vertex and the geocenter in seconds.
#[pyfunction]
pub fn time_delay_from_geocenter(vertex: [f64; 3], ra: f64, dec: f64, gps_time: f64) -> f64 {
    _time_delay_from_vertices(vertex.into(), GEOCENTER, ra, dec, gps_time)
}

/// Calculate the vecetor connecting the beam splitter to the end test mass.
/// 
/// # Arguments
/// 
/// * `arm_tilt` - The tilt angle of the arm in radians.
/// * `arm_azimuth` - The azimuth angle of the arm in radians.
/// * `longitude` - The longitude of the detector in radians.
/// * `latitude` - The latitude of the detector in radians.
/// 
/// # Returns
/// 
/// A `numpy` array representing the arm vector.
#[pyfunction]
pub fn calculate_arm(
    arm_tilt: f64,
    arm_azimuth: f64,
    longitude: f64,
    latitude: f64,
) -> Py<PyArray1<f64>> {
    let vec1: ThreeVector = SphericalAngles {
        zenith: -latitude,
        azimuth: longitude,
    }
    .into();
    let vec2: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2,
        azimuth: FRAC_PI_2 + longitude,
    }
    .into();
    let vec3: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2 - latitude,
        azimuth: longitude,
    }
    .into();
    let arm_vector: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2 - arm_tilt,
        azimuth: FRAC_PI_2 - arm_azimuth,
    }
    .into();
    ThreeMatrix::from_columns([vec1, vec2, vec3])
        .dot(arm_vector)
        .into()
}

/// Calculate the detector tensor for a given set of arm vectors.
/// 
/// # Arguments
/// 
/// * `arm_1` - The first arm vector as a 3-element array.
/// * `arm_2` - The second arm vector as a 3-element array.
/// 
/// # Returns
/// 
/// A `numpy` array representing the detector tensor.
#[pyfunction]
pub fn detector_tensor(x: [f64; 3], y: [f64; 3]) -> Py<PyArray2<f64>> {
    let det = DetectorGeometry::new(x.into(), y.into(), 1.0);
    det.detector_tensor.into()
}

/// A vectorized version of [`time_delay_from_geocenter`].
///
/// # Arguments
///
/// * `vertex` - The vertex as a 3-element array.
/// * `ra` - The right ascension of the source in radians.
/// * `dec` - The declination of the source in radians.
/// * `gps_times` - A list of GPS times.
///
/// # Returns
///
/// A `numpy` array of time delays in seconds.
///
/// [`time_delay_from_geocenter`]: ./fn.time_delay_from_geocenter.html
#[pyfunction]
pub fn time_delay_from_geocenter_vectorized(
    vertex: [f64; 3],
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
) -> Py<PyArray1<f64>> {
    let times = gps_times
        .iter()
        .map(|&gps_time| time_delay_from_geocenter(vertex, ra, dec, gps_time))
        .collect();
    Python::with_gil(|py| PyArray1::from_vec_bound(py, times).unbind())
}

/// Calculate the detector tensor for a set of GPS times.
///
/// See [`polarization_tensor`] for more details.
///
/// # Arguments
///
/// * `ra` - The right ascension of the source in radians.
/// * `dec` - The declination of the source in radians.
/// * `gps_times` - A list of GPS times (`shape=(N,)`).
/// * `psi` - The polarization angle of the source.
/// * `mode` - The polarization mode to calculate. This can be one of `'plus'`, `'cross'`,
/// `'longitudinal'`, `'breathing'`, `'x'`, or `'y'`.
///
/// # Returns
///
/// A `numpy` array of detector tensors (`shape=(N,3,3)`).
///
/// [`polarization_tensor`]: ./fn.polarization_tensor.html
#[pyfunction]
pub fn time_dependent_polarization_tensor(
    ra: f64,
    dec: f64,
    gps_times: Vec<f64>,
    psi: f64,
    mode: &str,
) -> Py<PyArray3<f64>> {
    let output: Vec<Vec<Vec<f64>>> = gps_times
        .iter()
        .map(|&gps_time| polarization_tensor(ra, dec, gps_time, psi, mode).into())
        .collect();
    Python::with_gil(|py| PyArray3::from_vec3_bound(py, &output).unwrap().unbind())
}
