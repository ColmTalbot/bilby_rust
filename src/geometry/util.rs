use std::f64::consts::PI;
use std::ops::{Add, Sub, Div, Mul};

use num_complex::Complex;
use pyo3::{Py, Python, pyfunction};
use numpy::{PyArray1, PyArray2};

use crate::time;

pub struct ThreeVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct ComplexThreeVector {
    pub x: Complex<f64>,
    pub y: Complex<f64>,
    pub z: Complex<f64>,
}

pub struct ThreeMatrix {
    pub rows: [ThreeVector; 3],
}

pub struct ComplexThreeMatrix {
    pub rows: [ComplexThreeVector; 3],
}

impl ThreeVector {
    pub fn from_array(array: &[f64; 3]) -> Self {
        Self {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }

    pub fn from_spherical_angles(theta: f64, phi: f64) -> Self {
        Self {
            x: theta.sin() * phi.cos(),
            y: theta.sin() * phi.sin(),
            z: theta.cos(),
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    fn norm(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn outer(&self, other: &Self) -> ThreeMatrix {
        ThreeMatrix {
            rows: [
                ThreeVector {
                    x: self.x * other.x,
                    y: self.x * other.y,
                    z: self.x * other.z,
                },
                ThreeVector {
                    x: self.y * other.x,
                    y: self.y * other.y,
                    z: self.y * other.z,
                },
                ThreeVector {
                    x: self.z * other.x,
                    y: self.z * other.y,
                    z: self.z * other.z,
                },
            ]
        }
    }

    pub fn to_pyarray(&self) -> Py<PyArray1<f64>> {
        let output = vec![self.x, self.y, self.z];
        Python::with_gil(|py| {
            PyArray1::from_vec_bound(py, output).unbind()
        })
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.x, self.y, self.z]
    }

    fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

}

impl Clone for ThreeVector {
    fn clone(&self) -> Self {
        Self {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl Copy for ThreeVector {}

impl Clone for ComplexThreeVector {
    fn clone(&self) -> Self {
        Self {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl Copy for ComplexThreeVector {}

impl Clone for ThreeMatrix {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows,
        }
    }
}

impl Copy for ThreeMatrix {}

impl Clone for ComplexThreeMatrix {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows,
        }
    }
}

impl Copy for ComplexThreeMatrix {}

impl Add for ThreeVector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ThreeVector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for ThreeVector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ThreeVector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f64> for ThreeVector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        ThreeVector {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Div<f64> for ThreeVector {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        ThreeVector {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl ThreeMatrix {

    pub fn to_array(&self) -> [[f64; 3]; 3] {
        [
            self.rows[0].to_array(),
            self.rows[1].to_array(),
            self.rows[2].to_array(),
        ]
    }

    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        self.rows.iter().map(|row| row.to_vec()).collect()
    }

    pub fn to_pyarray(&self) -> Py<PyArray2<f64>> {
        let output: Vec<Vec<f64>> = self.rows.iter().map(|row| vec![row.x, row.y, row.z]).collect();
        Python::with_gil(|py| {
            PyArray2::from_vec2_bound(py, &output).unwrap().unbind()
        })
    }

    pub fn iter(&self) -> std::slice::Iter<ThreeVector> {
        self.rows.iter()
    }

    pub fn dot(&self, other: &ThreeVector) -> ThreeVector {
        ThreeVector {
            x: self.rows[0].dot(other),
            y: self.rows[1].dot(other),
            z: self.rows[2].dot(other),
        }
    }
}

impl Add for ThreeMatrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ThreeMatrix {
            rows: [
                self.rows[0] + other.rows[0],
                self.rows[1] + other.rows[1],
                self.rows[2] + other.rows[2],
            ]
        }
    }
}

impl Sub for ThreeMatrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ThreeMatrix {
            rows: [
                self.rows[0] - other.rows[0],
                self.rows[1] - other.rows[1],
                self.rows[2] - other.rows[2],
            ]
        }
    }
}

impl Div<f64> for ThreeMatrix {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        ThreeMatrix {
            rows: [
                self.rows[0] / scalar,
                self.rows[1] / scalar,
                self.rows[2] / scalar,
            ]
        }
    }
}

impl Mul<Complex<f64>> for ThreeVector {
    type Output = ComplexThreeVector;

    fn mul(self, scalar: Complex<f64>) -> ComplexThreeVector {
        ComplexThreeVector {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<Complex<f64>> for &ThreeMatrix {
    type Output = ComplexThreeMatrix;

    fn mul(self, scalar: Complex<f64>) -> ComplexThreeMatrix {
        ComplexThreeMatrix {
            rows: [
                self.rows[0] * scalar,
                self.rows[1] * scalar,
                self.rows[2] * scalar,
            ]
        }
    }
}

impl Sub for ComplexThreeVector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ComplexThreeVector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub for ComplexThreeMatrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ComplexThreeMatrix {
            rows: [
                self.rows[0] - other.rows[0],
                self.rows[1] - other.rows[1],
                self.rows[2] - other.rows[2],
            ]
        }
    }
}

impl Mul<ThreeVector> for ComplexThreeVector {
    type Output = Self;

    fn mul(self, other: ThreeVector) -> Self {
        ComplexThreeVector {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<ThreeMatrix> for ComplexThreeMatrix {
    type Output = Self;

    fn mul(self, other: ThreeMatrix) -> Self {
        ComplexThreeMatrix {
            rows: [
                self.rows[0] * other.rows[0],
                self.rows[1] * other.rows[1],
                self.rows[2] * other.rows[2],
            ]
        }
    }
}

impl ComplexThreeVector {
    fn to_vec(&self) -> Vec<Complex<f64>> {
        vec![self.x, self.y, self.z]
    }
}

impl ComplexThreeMatrix {
    pub fn to_vec(&self) -> Vec<Vec<Complex<f64>>> {
        self.rows.iter().map(|row| row.to_vec()).collect()
    }

    pub fn sum(&self) -> Complex<f64> {
        self.rows.iter().map(|row| row.x + row.y + row.z).sum()
    }
}

#[pyfunction]
pub fn ra_dec_to_theta_phi(ra: f64, dec: f64, gps_time: f64) -> (f64, f64) {
    let gmst = time::greenwich_mean_sidereal_time(gps_time) % (2.0 * PI);
    let theta = PI / 2.0 - dec;
    let phi = ra - gmst;
    (theta, phi)
}
