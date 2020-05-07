use crate::error::{handle_error, IntegrationRetcode};
use crate::qk::fixed_order_gauss_kronrod;
use crate::utils::subinterval_too_small;
use crate::workspace::IntegrationWorkSpace;

#[allow(clippy::too_many_arguments)]
pub fn qag<F>(
    f: F,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    key: u8,
) -> std::result::Result<(f64, f64), IntegrationRetcode>
where
    F: Fn(f64) -> f64,
{
    // Roundoff detection counters
    let mut roundoff_type1: usize = 0;
    let mut roundoff_type2: usize = 0;

    // Integration workspace for storing results, intervals and errors
    let mut workspace = IntegrationWorkSpace::new(limit);
    workspace.alist[0] = a;
    workspace.blist[0] = b;

    // Initialize results
    let mut result = 0.0;
    let mut abserr = 0.0;

    // Check the tolerances
    if epsabs <= 0.0 && (epsrel < 50.0 * f64::EPSILON || epsrel < 0.5e-28) {
        return handle_error(result, abserr, IntegrationRetcode::BadTol);
    }

    // Perform the first integration
    let mut resabs = 0.0;
    let mut resasc = 0.0;
    result = fixed_order_gauss_kronrod(&f, a, b, key, &mut abserr, &mut resabs, &mut resasc);
    workspace.rlist[0] = result;
    workspace.elist[0] = abserr;
    workspace.size = 1;

    // Test on accuracy
    let mut tolerance = epsabs.max(epsrel * result.abs());
    // need IEEE rounding here to match original quadpack behavior
    // NOTE: GSL uses a volitile variable for extended precision registers. Should we do the same?
    let round_off = 50.0 * f64::EPSILON * resabs;

    if abserr <= round_off && abserr > tolerance {
        return handle_error(result, abserr, IntegrationRetcode::RoundOffFirstIter);
    } else if (abserr <= tolerance && abserr != resasc) || abserr.abs() < f64::EPSILON {
        return Ok((result, abserr));
    } else if limit == 1 {
        return handle_error(result, abserr, IntegrationRetcode::OneIterNotEnough);
    }

    let mut area = result;
    let mut errsum = abserr;

    let mut error_type: IntegrationRetcode = IntegrationRetcode::Success;

    let mut iter: usize = 1;
    loop {
        let (a_i, b_i, r_i, e_i) = workspace.recieve();

        let a1 = a_i;
        let b1 = 0.5 * (a_i + b_i);
        let a2 = b1;
        let b2 = b_i;

        let mut error1 = 0.0;
        let mut resabs1 = 0.0;
        let mut resasc1 = 0.0;
        let area1 =
            fixed_order_gauss_kronrod(&f, a1, b1, key, &mut error1, &mut resabs1, &mut resasc1);

        let mut error2 = 0.0;
        let mut resabs2 = 0.0;
        let mut resasc2 = 0.0;
        let area2 =
            fixed_order_gauss_kronrod(&f, a2, b2, key, &mut error2, &mut resabs2, &mut resasc2);

        let area12 = area1 + area2;
        let error12 = error1 + error2;

        errsum += error12 - e_i;
        area += area12 - r_i;

        if resasc1 != error1 && resasc2 != error2 {
            let delta = r_i - area12;

            if delta.abs() <= 1e-5 * area12.abs() && error12 >= 0.99 * e_i {
                roundoff_type1 += 1;
            }
            if iter >= 10 && error12 > e_i {
                roundoff_type2 += 1;
            }
        }

        tolerance = epsabs.max(epsrel * area.abs());

        if errsum > tolerance {
            if roundoff_type1 >= 6 || roundoff_type2 >= 20 {
                error_type = IntegrationRetcode::RoundOff;
            }

            // set error flag in the case of bad integrand behaviour at a point of the integration range
            if subinterval_too_small(a1, a2, b2) {
                error_type = IntegrationRetcode::BadIntegrand;
            }
        }

        workspace.update((a1, b1, area1, error1), (a2, b2, area2, error2));

        iter += 1;
        if iter >= limit || error_type != IntegrationRetcode::Success || errsum <= tolerance {
            break;
        }
    }

    result = workspace.sum_results();
    abserr = errsum;

    if errsum <= tolerance {
        error_type = IntegrationRetcode::Success;
    } else if iter == limit {
        error_type = IntegrationRetcode::TooManyIters;
    }

    handle_error(result, abserr, error_type)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::test_rel;

    fn f1(x: f64, alpha: f64) -> f64 {
        x.powf(alpha) * x.recip().ln()
    }

    fn f3(x: f64, alpha: f64) -> f64 {
        (2f64.powf(alpha) * x.sin()).cos()
    }

    fn f16(x: f64, alpha: f64) -> f64 {
        if x == 0.0 && alpha == 1.0 {
            1.0
        } else if x == 0.0 && alpha > 1.0 {
            0.0
        } else {
            x.powf(alpha - 1.0) * (1.0 + 10.0 * x).recip().powi(2)
        }
    }

    #[test]
    fn test_smooth_15() {
        let alpha = 2.6;
        let f = |x| f1(x, alpha);

        let exp_result = 7.716049382715854665E-02;
        let exp_abserr = 6.679384885865053037E-12;

        let result = qag(f, 0.0, 1.0, 0.0, 1e-10, 1000, 1);

        match result {
            Ok(result) => {
                test_rel(result.0, exp_result, 1e-15);
                test_rel(result.1, exp_abserr, 1e-6);
            }
            Err(err) => panic!("{:?}", err),
        }

        let result = qag(f, 1.0, 0.0, 0.0, 1e-10, 1000, 1);

        match result {
            Ok(result) => {
                test_rel(result.0, -exp_result, 1e-15);
                test_rel(result.1, exp_abserr, 1e-6);
            }
            Err(err) => panic!("{:?}", err),
        }
    }

    #[test]
    fn test_roundoff_panic() {
        let alpha = 1.3;
        let f = |x| f3(x, alpha);

        let result = qag(f, 0.3, 2.71, 1e-14, 0.0, 1000, 3);

        match result {
            Ok(_) => {
                assert!(false);
            }
            Err(err) => assert!(
                err == IntegrationRetcode::RoundOff || err == IntegrationRetcode::RoundOffFirstIter
            ),
        }
    }

    #[test]
    fn test_singularity_detection() {
        let alpha = 2.0;
        let f = |x| f16(x, alpha);

        let result = qag(f, -1.0, 1.0, 1e-14, 0.0, 1000, 5);

        match result {
            Ok(_) => {
                assert!(false);
            }
            Err(err) => assert!(err == IntegrationRetcode::BadIntegrand),
        }
    }
    #[test]
    fn test_iter_limit() {
        let alpha = 1.0;
        let f = |x| f16(x, alpha);

        let result = qag(f, -1.0, 1.0, 1e-14, 0.0, 1000, 6);

        match result {
            Ok(_) => {
                assert!(false);
            }
            Err(err) => assert!(err == IntegrationRetcode::TooManyIters),
        }
    }
}
