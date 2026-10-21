/// # bilby_rust
///
/// `bilby_rust` provides implementations of gravitational-wave detector geometry.
///
/// This is primarily intended to be used as a dependency for the `bilby` gravitational-wave
/// parameter estimation package. However, it can also be used as a standalone package for
/// computing detector responses and antenna patterns.
pub mod geometry;
pub mod time;

#[cfg(feature = "python")]
mod python;
