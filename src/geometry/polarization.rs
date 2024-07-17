use crate::geometry::util::{add_three_by_three, cross_product, outer, ra_dec_to_theta_phi};

fn m_vector(theta: f64, phi: f64, psi: f64) -> [f64; 3] {
    [
        -theta.cos() * phi.cos() * psi.sin() + phi.sin() * psi.cos(),
        -theta.cos() * phi.sin() * psi.sin() - phi.cos() * psi.cos(),
        theta.sin() * psi.sin(),
    ]
}

fn n_vector(theta: f64, phi: f64, psi: f64) -> [f64; 3] {
    [
        -theta.cos() * phi.cos() * psi.cos() - phi.sin() * psi.sin(),
        -theta.cos() * phi.sin() * psi.cos() + phi.cos() * psi.sin(),
        theta.sin() * psi.cos(),
    ]
}

fn omega_vector(theta: f64, phi: f64, psi: f64) -> [f64; 3] {
    cross_product(&m_vector(theta, phi, psi), &n_vector(theta, phi, psi)) 
}

pub fn symmetric_mode(input_1: &[f64; 3], input_2: &[f64; 3]) -> [[f64; 3]; 3] {
    add_three_by_three(
        &outer(input_1, input_2),
        &outer(input_2, input_1),
        &std::ops::Add::add,
    )
}

pub fn plus(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3]{
    let m = m_vector(theta, phi, psi);
    let n = n_vector(theta, phi, psi);
    add_three_by_three(
        &outer(&m, &m),
        &outer(&n, &n),
        &std::ops::Sub::sub,
    )
}

pub fn cross(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3] {
    let m = m_vector(theta, phi, psi);
    let n = n_vector(theta, phi, psi);
    symmetric_mode(&m, &n)
}

pub fn breathing(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3]{
    let m = m_vector(theta, phi, psi);
    let n = n_vector(theta, phi, psi);
    add_three_by_three(
        &outer(&m, &m),
        &outer(&n, &n),
        &std::ops::Add::add,
    )
}

pub fn longitudinal(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3]{
    let omega = omega_vector(theta, phi, psi);
    outer(&omega, &omega)
}

pub fn x(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3] {
    let m = m_vector(theta, phi, psi);
    let omega = omega_vector(theta, phi, psi);
    symmetric_mode(&m, &omega)
}

pub fn y(theta: f64, phi: f64, psi: f64) -> [[f64; 3]; 3] {
    let n = n_vector(theta, phi, psi);
    let omega = omega_vector(theta, phi, psi);
    symmetric_mode(&n, &omega)
}

pub fn polarization_tensor(ra: f64, dec: f64, gps_time: f64, psi: f64, mode: &str) -> [[f64; 3]; 3] {
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

