use num::Float;

#[inline]
pub fn test_positivity<T: Float>(result: T, resabs: T) -> bool {
    result.abs() >= (T::one() - T::from(50).unwrap() * T::epsilon()) * resabs
}

#[inline]
pub fn subinterval_too_small<T: Float>(a1: T, a2: T, b2: T) -> bool {
    let e = T::epsilon();
    let u = T::min_positive_value();

    let tmp = (T::one() + T::from(100).unwrap() * e) * (a2.abs() + T::from(1000).unwrap() * u);

    a1.abs() <= tmp && b2.abs() <= tmp
}
