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
    assert!(status != 1, "[Test uses subnormal value]".to_string());
}

/// Function 1 for integration testing
pub(crate) fn f1(x: f64, alpha: f64) -> f64 {
    x.powf(alpha) * x.recip().ln()
}

/// Function 3 for integration testing
pub(crate) fn f3(x: f64, alpha: f64) -> f64 {
    (2f64.powf(alpha) * x.sin()).cos()
}

/// Function 16 for integration testing
pub(crate) fn f11(x: f64, alpha: f64) -> f64 {
    x.recip().ln().powf(alpha - 1.0)
}

/// Function 15 for integration testing
pub(crate) fn f15(x: f64, alpha: f64) -> f64 {
    x * x * (-2f64.powf(-alpha) * x).exp()
}

/// Function 16 for integration testing
pub(crate) fn f16(x: f64, alpha: f64) -> f64 {
    if x == 0.0 && alpha == 1.0 {
        1.0
    } else if x == 0.0 && alpha > 1.0 {
        0.0
    } else {
        x.powf(alpha - 1.0) * (1.0 + 10.0 * x).recip().powi(2)
    }
}

pub(crate) fn f455(x: f64) -> f64 {
    x.ln() / (1.0 + 100.0 * x * x)
}
