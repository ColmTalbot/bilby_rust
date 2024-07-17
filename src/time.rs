use std::f64::consts::PI;

use::chrono::Datelike;
use::chrono::Timelike;
use::chrono::DateTime;
use::chrono::TimeDelta;
use::chrono::TimeZone;
use::chrono::Utc;

use pyo3::{Py, Python, pyfunction};
use numpy::PyArray1;


const LEAP_SECONDS: [i32; 18] = [
    46828800,
    78364801,
    109900802,
    173059203,
    252028804,
    315187205,
    346723206,
    393984007,
    425520008,
    457056009,
    504489610,
    551750411,
    599184012,
    820108813,
    914803214,
    1025136015,
    1119744016,
    1167264017,
];
const NUM_LEAPS: usize = 18;
const EPOCH_J2000_0_JD: f64 = 2451545.0;
const DAYS_PER_CENTURY: f64 = 36525.0;
const SECONDS_PER_DAY: f64 = 86400.0;
const SECONDS_PER_CENTURY: f64 = DAYS_PER_CENTURY * SECONDS_PER_DAY;

#[pyfunction]
pub fn n_leap_seconds(s: i32) -> i32 {
    let mut i: usize = NUM_LEAPS;
    while i > 0 && LEAP_SECONDS[i - 1] > s {
        i -= 1;
    }
    let i: i32 = i.try_into().unwrap();
    i
}


#[pyfunction]
pub fn gps_time_to_utc(gps_time: i32) -> DateTime<Utc> {
    let leap_seconds = n_leap_seconds(gps_time);
    let gps_epoch: DateTime<Utc> = Utc
        .with_ymd_and_hms(1980, 1, 6, 0, 0, 0)
        .unwrap();
    let gps_time = gps_epoch + TimeDelta::seconds(gps_time as i64);
    gps_time - TimeDelta::seconds(leap_seconds as i64)
}


#[pyfunction]
pub fn utc_to_julian_day(time: DateTime<Utc>) -> f64 {
    let year = time.year();
    let month = time.month() as i32;
    let day = time.day() as i32;
    let hour = time.hour() as i32;
    let minute = time.minute() as i32;
    let second = time.second() as i32;
    let julian_day: i32 = 
        367 * year
        - ((7 * (year + ((month + 9) / 12))) / 4)
        + ((275 * month) / 9)
        + day
        + 1721014;
    let seconds = hour * 3600 + minute * 60 + second;
    let fractional_day:f64 = seconds as f64 / SECONDS_PER_DAY - 0.5;
    let julian_day: f64 = julian_day as f64 + fractional_day;
    julian_day
}

#[pyfunction]
pub fn greenwich_sidereal_time(gps_time: f64, equation_of_equinoxes: f64) -> f64 {
    let julian_day = utc_to_julian_day(gps_time_to_utc(gps_time.floor() as i32));
    
    let t_high = (julian_day - EPOCH_J2000_0_JD) / DAYS_PER_CENTURY;
    let t_low = (gps_time - gps_time.floor()) / SECONDS_PER_CENTURY;
    let t = t_high + t_low;
    
    let mut sidereal_time = equation_of_equinoxes
        + (-6.2e-6 * t + 0.093104) * t.powf(2.0)
        + 67310.54841;
    sidereal_time += 8640184.812866 * t_high;
    sidereal_time += 3155760000.0 * t_high;
    sidereal_time += 8640184.812866 * t_low;
    sidereal_time += 3155760000.0 * t_low;

    sidereal_time * PI / 43200.0
}


#[pyfunction]
pub fn greenwich_mean_sidereal_time(gps_time: f64) -> f64 {
    greenwich_sidereal_time(gps_time, 0.0)
}


#[allow(dead_code)]
#[pyfunction]
pub fn greenwich_mean_sidereal_time_vectorized(gps_times: Vec<f64>) -> Py<PyArray1<f64>> {
    let times = gps_times.iter().map(
        |&gps_time| greenwich_mean_sidereal_time(gps_time)
    ).collect();
    Python::with_gil(|py| {
        PyArray1::from_vec_bound(py, times).unbind()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_leap_seconds() {
        assert_eq!(n_leap_seconds(1167264017), 18);
    }

    #[test]
    fn test_gps_time_to_utc() {
        assert_eq!(
            gps_time_to_utc(1167264017),
            Utc.with_ymd_and_hms(2016, 12, 31, 23, 59, 59).unwrap()
        );
    }

    #[test]
    fn test_utc_to_julian_day() {
        assert_eq!(
            utc_to_julian_day(Utc.with_ymd_and_hms(2016, 12, 31, 23, 59, 59).unwrap()),
            2457754.499988426
        );
    }

    #[test]
    fn test_greenwich_sidereal_time() {
        assert_eq!(greenwich_sidereal_time(1167264017.0, 0.0), 39127.15478913444);
    }
}

