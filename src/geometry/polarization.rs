use std::f64::consts::FRAC_PI_2;

use super::{
    three::{SphericalAngles, ThreeMatrix, ThreeVector},
    util::ra_dec_to_theta_phi,
};

/// The tensor modes that are available: `plus`, `cross`
pub const TENSOR_MODES: [&str; 2] = ["plus", "cross"];
/// The polarization modes that are available: `plus`, `cross`, `breathing`, `longitudinal`, `x`, `y`
pub const ALL_MODES: [&str; 6] = ["plus", "cross", "breathing", "longitudinal", "x", "y"];

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
            zenith: theta,
            azimuth: phi,
        }
        .into();
        let matrix = ThreeMatrix {
            rows: [vec1, vec2, -vec3],
        }
        .rotate_z(-psi);
        Self {
            m: matrix.rows[0],
            n: matrix.rows[1],
            omega: matrix.rows[2],
        }
    }

    fn plus(&self) -> ThreeMatrix {
        self.m.outer(self.m) - self.n.outer(self.n)
    }

    fn cross(&self) -> ThreeMatrix {
        symmetric_mode(self.m, self.n)
    }

    fn breathing(&self) -> ThreeMatrix {
        self.m.outer(self.m) + self.n.outer(self.n)
    }

    fn longitudinal(&self) -> ThreeMatrix {
        self.omega.outer(self.omega)
    }

    fn x(&self) -> ThreeMatrix {
        symmetric_mode(self.m, self.omega)
    }

    fn y(&self) -> ThreeMatrix {
        symmetric_mode(self.n, self.omega)
    }

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

pub fn polarization_tensor(ra: f64, dec: f64, gps_time: f64, psi: f64, mode: &str) -> ThreeMatrix {
    let (theta, phi) = ra_dec_to_theta_phi(ra, dec, gps_time);
    PolarizationMatrix::new(theta, phi, psi).mode(mode)
}
