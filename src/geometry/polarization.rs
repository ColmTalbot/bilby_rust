use std::f64::consts::{FRAC_PI_2, PI};

use super::{
    three::{SphericalAngles, ThreeMatrix, ThreeVector},
    util::ra_dec_to_theta_phi,
};

/// The tensor modes that are available: `plus`, `cross`
pub const TENSOR_MODES: [&str; 2] = ["plus", "cross"];
/// The polarization modes that are available: `plus`, `cross`, `breathing`, `longitudinal`, `x`, `y`
pub const ALL_MODES: [&str; 6] = ["plus", "cross", "breathing", "longitudinal", "x", "y"];

/// A struct representing the polarization matrix to construct the antenna response to the various modes.
///
/// The polarization matrix is defined by three angles (`\theta`, `\phi`, `\psi`) which are then
/// used to construct orthogonal unit vectors: `m`, `n`, and `omega`.
#[derive(Debug)]
pub struct PolarizationMatrix {
    m: ThreeVector,
    n: ThreeVector,
    omega: ThreeVector,
}

impl PolarizationMatrix {
    pub fn new(theta: f64, phi: f64, psi: f64) -> Self {
        let vec1: ThreeVector = SphericalAngles {
            zenith: FRAC_PI_2,
            azimuth: phi - FRAC_PI_2,
        }
        .into();
        let vec2: ThreeVector = SphericalAngles {
            zenith: theta - FRAC_PI_2,
            azimuth: phi,
        }
        .into();
        let vec3: ThreeVector = SphericalAngles {
            zenith: PI - theta,
            azimuth: PI + phi,
        }
        .into();
        let matrix = ThreeMatrix {
            rows: [vec1, vec2, vec3],
        }
        .rotate_z(-psi);
        Self {
            m: matrix.rows[0],
            n: matrix.rows[1],
            omega: matrix.rows[2],
        }
    }

    /// The plus polarization response.
    ///
    /// F_{+} = m \otimes m - n \otimes n
    pub fn plus(&self) -> ThreeMatrix {
        self.m.outer(self.m) - self.n.outer(self.n)
    }

    /// The cross polarization response.
    ///
    /// F_{\cross} = m \otimes n + n \otimes m
    pub fn cross(&self) -> ThreeMatrix {
        symmetric_mode(self.m, self.n)
    }

    /// The breathing polarization response.
    ///
    /// F_{b} = m \otimes m + n \otimes n
    pub fn breathing(&self) -> ThreeMatrix {
        self.m.outer(self.m) + self.n.outer(self.n)
    }

    /// The longitudinal polarization response.
    ///
    /// F_{l} = \omega \otimes \omega
    pub fn longitudinal(&self) -> ThreeMatrix {
        self.omega.outer(self.omega)
    }

    /// The vector x polarization response.
    ///
    /// F_{x} = m \otimes \omega + \omega \otimes m
    pub fn x(&self) -> ThreeMatrix {
        symmetric_mode(self.m, self.omega)
    }

    /// The vector y polarization response.
    ///
    /// F_{y} = n \otimes \omega + \omega \otimes n
    pub fn y(&self) -> ThreeMatrix {
        symmetric_mode(self.n, self.omega)
    }

    /// A convenience method to get the polarization mode based on the input string.
    pub fn mode(&self, mode: &str) -> ThreeMatrix {
        match mode {
            "plus" => self.plus(),
            "cross" => self.cross(),
            "breathing" => self.breathing(),
            "longitudinal" => self.longitudinal(),
            "x" => self.x(),
            "y" => self.y(),
            _ => panic!("{mode} not a polarization mode!"),
        }
    }
}

fn symmetric_mode(input_1: ThreeVector, input_2: ThreeVector) -> ThreeMatrix {
    input_1.outer(input_2) + input_2.outer(input_1)
}

/// Get the polarization tensor for a given right ascension, declination, GPS time, and polarization angle.
///
/// # Arguments
///
/// * `ra` - Right ascension in radians.
/// * `dec` - Declination in radians.
/// * `gps_time` - GPS time in seconds since the GPS epoch (January 6, 1980).
/// * `psi` - Polarization angle in radians.
/// * `mode` - The polarization mode to calculate.
///
/// # Returns
///
/// A `ThreeMatrix` representing the polarization tensor.
pub fn polarization_tensor(ra: f64, dec: f64, gps_time: f64, psi: f64, mode: &str) -> ThreeMatrix {
    let (theta, phi) = ra_dec_to_theta_phi(ra, dec, gps_time);
    PolarizationMatrix::new(theta, phi, psi).mode(mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_matrix_approx_eq {
        ($a:expr, $b:expr) => {
            const EPSILON: f64 = 1e-10;
            for i in 0..3 {
                assert!(($a.rows[i].x - $b.rows[i].x).abs() < EPSILON);
                assert!(($a.rows[i].y - $b.rows[i].y).abs() < EPSILON);
                assert!(($a.rows[i].z - $b.rows[i].z).abs() < EPSILON);
            }
        };
    }

    #[test]
    fn test_plus_mode() {
        let pm = PolarizationMatrix::new(1.0, 2.0, 3.0);
        let result = pm.plus();
        println!("{:?}", result);
        let expected = ThreeMatrix {
            rows: [
                ThreeVector {
                    x: 0.6310944283106209,
                    y: 0.5680750305371688,
                    z: -0.3954588595258747,
                },
                ThreeVector {
                    x: 0.5680750305371688,
                    y: 0.04877662868261845,
                    z: 0.29910040882446337,
                },
                ThreeVector {
                    x: -0.3954588595258747,
                    y: 0.29910040882446337,
                    z: -0.6798710569932394,
                },
            ],
        };
        assert_matrix_approx_eq!(result, expected);
    }

    #[test]
    fn test_cross_mode() {
        let pm = PolarizationMatrix::new(1.0, 2.0, 3.0);
        let result = pm.cross();
        println!("{:?}", result);
        let expected = ThreeMatrix {
            rows: [
                ThreeVector {
                    x: 0.6095165675666464,
                    y: -0.20250177557045979,
                    z: 0.6818062130067495,
                },
                ThreeVector {
                    x: -0.20250177557045979,
                    y: -0.41166988063832005,
                    z: 0.45174151309178,
                },
                ThreeVector {
                    x: 0.6818062130067495,
                    y: 0.45174151309178,
                    z: -0.19784668692832622,
                },
            ],
        };
        assert_matrix_approx_eq!(result, expected);
    }

    #[test]
    fn test_breathing_mode() {
        let pm = PolarizationMatrix::new(1.0, 2.0, 3.0);
        let result = pm.breathing();
        println!("{:?}", result);
        let expected = ThreeMatrix {
            rows: [
                ThreeVector {
                    x: 0.8773771273420202,
                    y: 0.2679358649053266,
                    z: 0.1892006238269821,
                },
                ThreeVector {
                    x: 0.2679358649053266,
                    y: 0.41454945438440843,
                    z: -0.4134109052159029,
                },
                ThreeVector {
                    x: 0.1892006238269821,
                    y: -0.4134109052159029,
                    z: 0.7080734182735711,
                },
            ],
        };

        assert_matrix_approx_eq!(result, expected);
    }

    #[test]
    fn test_longitudinal_mode() {
        let pm = PolarizationMatrix::new(1.0, 2.0, 3.0);
        let result = pm.longitudinal();
        println!("{:?}", result);
        let expected = ThreeMatrix {
            rows: [
                ThreeVector {
                    x: 0.12262287265797965,
                    y: -0.26793586490532656,
                    z: -0.18920062382698202,
                },
                ThreeVector {
                    x: -0.26793586490532656,
                    y: 0.5854505456155917,
                    z: 0.413410905215903,
                },
                ThreeVector {
                    x: -0.18920062382698202,
                    y: 0.413410905215903,
                    z: 0.2919265817264287,
                },
            ],
        };
        assert_matrix_approx_eq!(result, expected);
    }

    #[test]
    fn test_vector_x_mode() {
        let pm = PolarizationMatrix::new(1.0, 2.0, 3.0);
        let result = pm.x();
        println!("{:?}", result);
        let expected = ThreeMatrix {
            rows: [
                ThreeVector {
                    x: -0.6082320535403873,
                    y: 0.49596132203235593,
                    z: 0.5108177625907906,
                },
                ThreeVector {
                    x: 0.49596132203235593,
                    y: 0.7365521137428442,
                    z: 0.16919497019111168,
                },
                ThreeVector {
                    x: 0.5108177625907906,
                    y: 0.16919497019111168,
                    z: -0.12832006020245665,
                },
            ],
        };
        assert_matrix_approx_eq!(result, expected);
    }

    #[test]
    fn test_vector_y_mode() {
        let pm = PolarizationMatrix::new(1.0, 2.0, 3.0);
        let result = pm.y();
        println!("{:?}", result);
        let expected = ThreeMatrix {
            rows: [
                ThreeVector {
                    x: -0.24576367527033288,
                    y: 0.4182550352491018,
                    z: -0.10211348383083577,
                },
                ThreeVector {
                    x: 0.4182550352491018,
                    y: -0.6544339544651846,
                    z: 0.40634453755196465,
                },
                ThreeVector {
                    x: -0.10211348383083577,
                    y: 0.40634453755196465,
                    z: 0.9001976297355173,
                },
            ],
        };
        assert_matrix_approx_eq!(result, expected);
    }
}
