#[inline]
pub fn test_positivity(result: f64, resabs: f64) -> bool {
    result.abs() >= (1.0 - 50.0 * f64::EPSILON) * resabs
}

#[inline]
pub fn subinterval_too_small(a1: f64, a2: f64, b2: f64) -> bool {
    let e = f64::EPSILON;
    let u = f64::MIN_POSITIVE;

    let tmp = (1.0 + 100.0 * e) * (a2.abs() + 1000.0 * u);

    a1.abs() <= tmp && b2.abs() <= tmp
}
