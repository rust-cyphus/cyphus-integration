pub(crate) fn test_rel(result: f64, expected: f64, relative_error: f64) {
    let mut status: i32 = 0;
    // Check for nan or inf or number
    if result.is_nan() || expected.is_nan() {
        status = if result.is_nan() != expected.is_nan() {
            1
        } else {
            0
        };
    } else if result.is_infinite() || expected.is_infinite() {
        status = if result.is_infinite() != expected.is_infinite() {
            1
        } else {
            0
        };
    } else if (expected > 0.0 && expected < f64::MIN_POSITIVE)
        || (expected < 0.0 && expected > -(f64::MIN_POSITIVE))
    {
        status = -1;
    } else if expected != 0.0 {
        status = if (result - expected).abs() / expected.abs() > relative_error {
            1
        } else {
            0
        };
    }

    assert!(
        status == 0,
        format!("observed: {:?}, expected: {:?}", result, expected)
    );
    assert!(status != 1, format!("[Test uses subnormal value]"));
}
