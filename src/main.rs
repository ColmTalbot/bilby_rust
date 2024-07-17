mod time;
mod geometry;

fn main() {
    println!("{}", time::n_leap_seconds(1167264017));
    println!("{}", time::gps_time_to_utc(1167264017));
    println!("{}", time::utc_to_julian_day(time::gps_time_to_utc(1167264017)));
    println!("{}", time::greenwich_sidereal_time(1167264017.0, 0.0));
    println!("{}", time::greenwich_mean_sidereal_time(1167264017.0));

    println!("{:?}", geometry::zenith_azimuth_to_theta_phi_optimized(
        0.0,
        0.0,
        [-2106630.53831713, -3865062.66105881,  4600350.22663899],
        [   74276.0447238 , -5496283.71970837,  3224257.01743562],
    ));
}
