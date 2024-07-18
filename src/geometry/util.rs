use std::ops::{Add, Div, Mul, Sub};

use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::{Py, Python};

pub struct SphericalAngles {
    pub zenith: f64,
    pub azimuth: f64,
}

impl From<SphericalAngles> for ThreeVector {
    fn from(angles: SphericalAngles) -> Self {
        Self {
            x: angles.zenith.sin() * angles.azimuth.cos(),
            y: angles.zenith.sin() * angles.azimuth.sin(),
            z: angles.zenith.cos(),
        }
    }
}

impl From<(f64, f64)> for SphericalAngles {
    fn from(pair: (f64, f64)) -> Self {
        Self {
            zenith: pair.0,
            azimuth: pair.1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThreeVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ComplexThreeVector {
    pub x: Complex<f64>,
    pub y: Complex<f64>,
    pub z: Complex<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ThreeMatrix {
    pub rows: [ThreeVector; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct ComplexThreeMatrix {
    pub rows: [ComplexThreeVector; 3],
}

impl ThreeVector {
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn normalize(self) -> Self {
        let norm = self.norm();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    fn norm(self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn outer(self, other: Self) -> ThreeMatrix {
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
            ],
        }
    }
}

impl From<ThreeVector> for [f64; 3] {
    fn from(vector: ThreeVector) -> Self {
        [vector.x, vector.y, vector.z]
    }
}

impl From<&ThreeVector> for Vec<f64> {
    fn from(vector: &ThreeVector) -> Self {
        vec![vector.x, vector.y, vector.z]
    }
}

impl From<ThreeVector> for Py<PyArray1<f64>> {
    fn from(vector: ThreeVector) -> Self {
        let output = vec![vector.x, vector.y, vector.z];
        Python::with_gil(|py| PyArray1::from_vec_bound(py, output).unbind())
    }
}

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

impl From<[f64; 3]> for ThreeVector {
    fn from(array: [f64; 3]) -> Self {
        Self {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }
}

impl ThreeMatrix {
    pub fn iter(&self) -> std::slice::Iter<'_, ThreeVector> {
        self.rows.iter()
    }

    pub fn dot(self, other: ThreeVector) -> ThreeVector {
        ThreeVector {
            x: self.rows[0].dot(other),
            y: self.rows[1].dot(other),
            z: self.rows[2].dot(other),
        }
    }
}

impl From<ThreeMatrix> for [[f64; 3]; 3] {
    fn from(matrix: ThreeMatrix) -> Self {
        [
            matrix.rows[0].into(),
            matrix.rows[1].into(),
            matrix.rows[2].into(),
        ]
    }
}

impl From<ThreeMatrix> for Vec<Vec<f64>> {
    fn from(matrix: ThreeMatrix) -> Self {
        matrix.rows.iter().map(|row| row.into()).collect()
    }
}

impl From<ThreeMatrix> for Py<PyArray2<f64>> {
    fn from(matrix: ThreeMatrix) -> Self {
        let output: Vec<Vec<f64>> = matrix.into();
        Python::with_gil(|py| PyArray2::from_vec2_bound(py, &output).unwrap().unbind())
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
            ],
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
            ],
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
            ],
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
            ],
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
            ],
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
            ],
        }
    }
}

impl ComplexThreeVector {
    fn to_vec(self) -> Vec<Complex<f64>> {
        vec![self.x, self.y, self.z]
    }
}

impl ComplexThreeMatrix {
    pub fn to_vec(self) -> Vec<Vec<Complex<f64>>> {
        self.rows.iter().map(|row| row.to_vec()).collect()
    }

    pub fn sum(self) -> Complex<f64> {
        self.rows.iter().map(|row| row.x + row.y + row.z).sum()
    }
}
