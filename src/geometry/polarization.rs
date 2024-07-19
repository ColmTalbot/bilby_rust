use std::f64::consts::FRAC_PI_2;

use super::{
    three::{SphericalAngles, ThreeMatrix, ThreeVector},
    util::ra_dec_to_theta_phi,
};

fn m_vector(theta: f64, phi: f64, psi: f64) -> ThreeVector {
    let vec1: ThreeVector = SphericalAngles {
        zenith: theta - FRAC_PI_2,
        azimuth: phi,
    }
    .into();
    let vec2: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2,
        azimuth: phi - FRAC_PI_2,
    }
    .into();
    vec1 * psi.sin() + vec2 * psi.cos()
}

fn n_vector(theta: f64, phi: f64, psi: f64) -> ThreeVector {
    let vec1: ThreeVector = SphericalAngles {
        zenith: theta - FRAC_PI_2,
        azimuth: phi,
    }
    .into();
    let vec2: ThreeVector = SphericalAngles {
        zenith: FRAC_PI_2,
        azimuth: phi - FRAC_PI_2,
    }
    .into();
    vec1 * psi.cos() - vec2 * psi.sin()
}

fn omega_vector(theta: f64, phi: f64, psi: f64) -> ThreeVector {
    m_vector(theta, phi, psi).cross(n_vector(theta, phi, psi))
}

pub fn symmetric_mode(input_1: ThreeVector, input_2: ThreeVector) -> ThreeMatrix {
    input_1.outer(input_2) + input_2.outer(input_1)
}

pub fn plus(theta: f64, phi: f64, psi: f64) -> ThreeMatrix {
    let m = m_vector(theta, phi, psi);
    let n = n_vector(theta, phi, psi);
    m.outer(m) - n.outer(n)
}

pub fn cross(theta: f64, phi: f64, psi: f64) -> ThreeMatrix {
    let m = m_vector(theta, phi, psi);
    let n = n_vector(theta, phi, psi);
    symmetric_mode(m, n)
}

pub fn breathing(theta: f64, phi: f64, psi: f64) -> ThreeMatrix {
    let m = m_vector(theta, phi, psi);
    let n = n_vector(theta, phi, psi);
    m.outer(m) + n.outer(n)
}

pub fn longitudinal(theta: f64, phi: f64, psi: f64) -> ThreeMatrix {
    let omega = omega_vector(theta, phi, psi);
    omega.outer(omega)
}

pub fn x(theta: f64, phi: f64, psi: f64) -> ThreeMatrix {
    let m = m_vector(theta, phi, psi);
    let omega = omega_vector(theta, phi, psi);
    symmetric_mode(m, omega)
}

pub fn y(theta: f64, phi: f64, psi: f64) -> ThreeMatrix {
    let n = n_vector(theta, phi, psi);
    let omega = omega_vector(theta, phi, psi);
    symmetric_mode(n, omega)
}

pub fn polarization_tensor(ra: f64, dec: f64, gps_time: f64, psi: f64, mode: &str) -> ThreeMatrix {
    let theta_phi = ra_dec_to_theta_phi(ra, dec, gps_time);
    let theta = theta_phi.0;
    let phi = theta_phi.1;

    let func = match mode {
        "plus" => plus,
        "cross" => cross,
        "breathing" => breathing,
        "longitudinal" => longitudinal,
        "x" => x,
        "y" => y,
        _ => panic!("{mode} not a polarization mode!"),
    };
    func(theta, phi, psi)
}
