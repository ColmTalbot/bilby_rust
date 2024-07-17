# bilby-rust

[Rust](https://www.rust-lang.org/) implementation of domain-specific geometrical operations for [Bilby](https://git.ligo.org/lscsoft/bilby) to mirror the functionality in [bilby-cython](https://git.ligo.org/colm.talbot/bilby-cython) using [PyO3](https:/pyo3.rs).

## Usage

To compile the project and install the python bindings you can use [maturin](https://github.com/PyO3/maturin)

```bash
$ maturin develop
```

This will install an unoptimized version, if you want to test benchmarking, be sure to install with

```bash
$ maturin develop --release
```

The python code can then be used directly in python

```python
>>> from bilby_rust.time import greenwich_mean_sidereal_time
>>> greenwich_mean_sidereal_time(1e9)
26930.069103915423
```

The API doesn't completely match `bilby-cython`:
- there is currently no support for writing `numpy` `ufuncs` using `PyO3` so the vectorized version of `greenwich_mean_sidereal_time` is implemented as `bilby_rust.time.greenwich_mean_sidereal_time_vectorized`.
- additional functionality for dealing with time- and frequency-dependent antenna response functions is additionally implemented using, e.g., `bilby_rust.geometry.{time_dependent_polarization_tensor,time_delay_from_geocenter_vectorized,frequency_dependent_detector_tensor}`. This was translated from an [implementation](https://git.ligo.org/jacob.golomb/bilby-cython/-/tree/long_wavelength) by @jacobgolomb for `bilby-cython`.
- the convention for the detector-based reference frame from [Roulet+](https://arxiv.org/abs/2207.03508) is included as `bilby_rust.geometry.zenith_azimuth_to_theta_phi_optimized`.
- calculation of the antenna response that projects the detector pattern against the polarization tensors using `bilby_rust.geometry.{antenna_response,antenna_response_all_model,antenna_response_tensor_modes}`.