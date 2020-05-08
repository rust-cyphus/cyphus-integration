use crate::qk::fixed_order_gauss_kronrod;
use crate::utils::subinterval_too_small;
use crate::workspace::IntegrationWorkSpace;
use crate::result::{IntegrationResult, IntegrationRetCode};

#[allow(clippy::too_many_arguments)]
pub fn qag<F>(
    f: F,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    key: u8,
) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
{
    let mut result = IntegrationResult::new();

    // Roundoff detection counters
    let mut roundoff_type1: usize = 0;
    let mut roundoff_type2: usize = 0;

    // Integration workspace for storing results, intervals and errors
    let mut workspace = IntegrationWorkSpace::new(limit);
    workspace.alist[0] = a;
    workspace.blist[0] = b;


    // Check the tolerances
    if epsabs <= 0.0 && (epsrel < 50.0 * f64::EPSILON || epsrel < 0.5e-28) {
        result.code = IntegrationRetCode::BadTol;
        result.issue_warning(Some(&[epsabs, epsrel]));
        return result;
    }

    // Perform the first integration
    let (val, err, mut resabs, mut resasc) = fixed_order_gauss_kronrod(&f, a, b, key);
    result.val = val;
    result.err = err;
    workspace.rlist[0] = result.val;
    workspace.elist[0] = result.err;
    workspace.size = 1;

    // Test on accuracy
    let mut tolerance = epsabs.max(epsrel * result.val.abs());
    // need IEEE rounding here to match original QUADPACK behavior
    // NOTE: GSL uses a volatile variable for extended precision registers. Should we do the same?
    let round_off = 50.0 * f64::EPSILON * resabs;

    if result.err <= round_off && result.err > tolerance {
        result.code = IntegrationRetCode::RoundOffFirstIter;
        result.issue_warning(None);
        return result;
    } else if (result.err <= tolerance && result.err != resasc) || result.err.abs() < f64::EPSILON {
        return result;
    } else if limit == 1 {
        result.code = IntegrationRetCode::OneIterNotEnough;
        result.issue_warning(None);
        return result;
    }

    let mut area = result.val;
    let mut errsum = result.err;

    let mut iter: usize = 1;
    loop {
        let (a_i, b_i, r_i, e_i) = workspace.recieve();

        let a1 = a_i;
        let b1 = 0.5 * (a_i + b_i);
        let a2 = b1;
        let b2 = b_i;

        let (area1, error1, _, resasc1) = fixed_order_gauss_kronrod(&f, a1, b1, key);
        let (area2, error2, _, resasc2) = fixed_order_gauss_kronrod(&f, a2, b2, key);

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
                result.code = IntegrationRetCode::RoundOff;
            }

            // set error flag in the case of bad integrand behaviour at a point of the integration range
            if subinterval_too_small(a1, a2, b2) {
                result.code = IntegrationRetCode::BadIntegrand;
            }
        }

        workspace.update((a1, b1, area1, error1), (a2, b2, area2, error2));

        iter += 1;
        if iter >= limit || result.code != IntegrationRetCode::Success || errsum <= tolerance {
            break;
        }
    }

    result.val = workspace.sum_results();
    result.err = errsum;

    result.code = if errsum <= tolerance {
        IntegrationRetCode::Success
    } else if iter == limit {
        IntegrationRetCode::TooManyIters
    } else {
        result.code
    };

    result
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_smooth_15() {
        let alpha = 2.6;
        let f = |x| f1(x, alpha);

        let exp_result = 7.716049382715854665E-02;
        let exp_abserr = 6.679384885865053037E-12;

        let result = qag(f, 0.0, 1.0, 0.0, 1e-10, 1000, 1);

        test_rel(result.val, exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-6);

        let result = qag(f, 1.0, 0.0, 0.0, 1e-10, 1000, 1);

        test_rel(result.val, -exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-6);
    }

    #[test]
    fn test_roundoff_panic() {
        let alpha = 1.3;
        let f = |x| f3(x, alpha);

        let result = qag(f, 0.3, 2.71, 1e-14, 0.0, 1000, 3);

        assert!(result.code == IntegrationRetCode::RoundOff || result.code ==
            IntegrationRetCode::RoundOffFirstIter);
    }

    #[test]
    fn test_singularity_detection() {
        let alpha = 2.0;
        let f = |x| f16(x, alpha);

        let result = qag(f, -1.0, 1.0, 1e-14, 0.0, 1000, 5);

        assert!(result.code == IntegrationRetCode::BadIntegrand);
    }

    #[test]
    fn test_iter_limit() {
        let alpha = 1.0;
        let f = |x| f16(x, alpha);

        let result = qag(f, -1.0, 1.0, 1e-14, 0.0, 1000, 6);
        assert!(result.code == IntegrationRetCode::TooManyIters);
    }
}
