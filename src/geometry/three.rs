use std::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::{Py, Python};

/// Container for spherical angles
///
/// # Fields
///
/// * `zenith`: Zenith angle in radians
/// * `azimuth`: Azimuth angle in radians
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

/// Container for a three-dimensional vector of floats, to avoid
/// having to dynamically allocate memory for a `Vec<f64>` or
/// `Array1<f64>`
///
/// # Fields
///
/// * `x`: x-component of the vector
/// * `y`: y-component of the vector
/// * `z`: z-component of the vector
#[derive(Debug, Clone, Copy)]
pub struct ThreeVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Container for a three-dimensional vector of complex numbers, to avoid
/// having to dynamically allocate memory for a `Vec<Complex<f64>>` or
/// `Array1<Complex<f64>>`
///
///
/// # Fields
///
/// * `x`: x-component of the vector
/// * `y`: y-component of the vector
/// * `z`: z-component of the vector
#[derive(Debug, Clone, Copy)]
pub struct ComplexThreeVector {
    pub x: Complex<f64>,
    pub y: Complex<f64>,
    pub z: Complex<f64>,
}

/// Container for a three-by-three matrix of floats, to avoid
/// having to dynamically allocate memory for a `Vec<Vec<f64>>` or
/// `Array2<f64>`
///
/// # Fields
///
/// * `rows`: Array of three `ThreeVector` structs
#[derive(Debug, Clone, Copy)]
pub struct ThreeMatrix {
    pub rows: [ThreeVector; 3],
}

/// Container for a three-by-three matrix of complex numbers, to avoid
/// having to dynamically allocate memory for a `Vec<Vec<Complex<f64>>>` or
/// `Array2<Complex<f64>>`
///
/// # Fields
///
/// * `rows`: Array of three `ComplexThreeVector` structs
#[derive(Debug, Clone, Copy)]
pub struct ComplexThreeMatrix {
    pub rows: [ComplexThreeVector; 3],
}

impl ThreeVector {
    /// Calculate the dot product of two vectors
    ///
    /// # Arguments
    ///
    /// * `other`: The other vector
    ///
    /// # Returns
    ///
    /// The dot product of the two vectors
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculate the L2-norm of the vector
    ///
    /// # Returns
    ///
    /// The L2-norm of the vector
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

    /// Compute the cross product of two vectors
    ///
    /// # Arguments
    ///
    /// * `other`: The other vector
    ///
    /// # Returns
    ///
    /// The cross product of the two vectors
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Compute the element-wise outer product of two vectors
    ///
    /// # Arguments
    ///
    /// * `other`: The other vector
    ///
    /// # Returns
    ///
    /// The element-wise outer product of the two vectors
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

impl From<[f64; 3]> for ThreeVector {
    fn from(array: [f64; 3]) -> Self {
        Self {
            x: array[0],
            y: array[1],
            z: array[2],
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

/// Add two vectors element-wise
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

/// Subtract two vectors element-wise
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

/// Multiply by a scalar element-wise
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

/// Divide by a scalar element-wise
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

impl Neg for ThreeVector {
    type Output = Self;

    fn neg(self) -> Self {
        ThreeVector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Multiply by a complex scalar element-wise
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

impl ComplexThreeVector {
    fn to_vec(self) -> Vec<Complex<f64>> {
        vec![self.x, self.y, self.z]
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

impl ThreeMatrix {
    pub fn iter(&self) -> std::slice::Iter<'_, ThreeVector> {
        self.rows.iter()
    }

    pub fn from_columns(columns: [ThreeVector; 3]) -> Self {
        Self {
            rows: [
                ThreeVector {
                    x: columns[0].x,
                    y: columns[1].x,
                    z: columns[2].x,
                },
                ThreeVector {
                    x: columns[0].y,
                    y: columns[1].y,
                    z: columns[2].y,
                },
                ThreeVector {
                    x: columns[0].z,
                    y: columns[1].z,
                    z: columns[2].z,
                },
            ],
        }
    }

    pub fn dot(self, other: ThreeVector) -> ThreeVector {
        ThreeVector {
            x: self.rows[0].dot(other),
            y: self.rows[1].dot(other),
            z: self.rows[2].dot(other),
        }
    }

    pub fn sum(self) -> f64 {
        self.rows.iter().map(|row| row.x + row.y + row.z).sum()
    }

    pub fn transpose(self) -> Self {
        Self::from_columns(self.rows)
    }

    pub fn rotate_x(self, angle: f64) -> ThreeMatrix {
        ThreeMatrix {
            rows: [
                self.rows[0],
                self.rows[1] * angle.cos() - self.rows[2] * angle.sin(),
                self.rows[1] * angle.sin() + self.rows[2] * angle.cos(),
            ],
        }
    }

    pub fn rotate_y(self, angle: f64) -> ThreeMatrix {
        ThreeMatrix {
            rows: [
                self.rows[2] * angle.sin() + self.rows[0] * angle.cos(),
                self.rows[1],
                self.rows[2] * angle.cos() - self.rows[0] * angle.sin(),
            ],
        }
    }

    pub fn rotate_z(self, angle: f64) -> ThreeMatrix {
        ThreeMatrix {
            rows: [
                self.rows[0] * angle.cos() - self.rows[1] * angle.sin(),
                self.rows[0] * angle.sin() + self.rows[1] * angle.cos(),
                self.rows[2],
            ],
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

/// Multiply by a complex scalar element-wise
impl Mul<Complex<f64>> for ThreeMatrix {
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

/// Multiply by a vector, each row is multiplied by the corresponding element of the vector
///
/// **Note**: This is not an inner product, for that use the [`dot`] method
///
/// [`dot`]: struct.ThreeMatrix.html#method.dot
impl Mul<ThreeVector> for ThreeMatrix {
    type Output = ThreeMatrix;

    fn mul(self, other: ThreeVector) -> ThreeMatrix {
        ThreeMatrix {
            rows: [
                self.rows[0] * other.x,
                self.rows[1] * other.y,
                self.rows[2] * other.z,
            ],
        }
    }
}

impl ComplexThreeMatrix {
    pub fn to_vec(self) -> Vec<Vec<Complex<f64>>> {
        self.rows.iter().map(|row| row.to_vec()).collect()
    }

    /// Calculate the element-wise sum of the matrix, equivalent to a double contraction
    pub fn sum(self) -> Complex<f64> {
        self.rows.iter().map(|row| row.x + row.y + row.z).sum()
    }
}

/// Subtract two vectors element-wise
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

/// Subtract two matrices element-wise
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

/// Multiply by another matrix element-wise
///
/// **Note**: This is not standard matrix multiplication
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
