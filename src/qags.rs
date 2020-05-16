use crate::extrap::ExtrapolationTable;
use crate::qk::fixed_order_gauss_kronrod;
use crate::result::{IntegrationResult, IntegrationRetCode};
use crate::utils::{subinterval_too_small, test_positivity};
use crate::workspace::IntegrationWorkSpace;

#[allow(clippy::too_many_arguments)]
pub fn qags<F>(
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

    let mut workspace = IntegrationWorkSpace::new(limit);
    workspace.alist[0] = a;
    workspace.blist[0] = b;

    // Test on accuracy
    if epsabs <= 0.0 && (epsrel < 50.0 * f64::EPSILON || epsrel < f64::EPSILON) {
        result.code = IntegrationRetCode::BadTol;
        result.issue_warning(Some(&[epsabs, epsrel]));
        return result;
    }

    // Perform the first integration
    let (val, err, resabs, resasc) = fixed_order_gauss_kronrod(&f, a, b, key);
    result.val = val;
    result.err = err;

    workspace.rlist[0] = result.val;
    workspace.elist[0] = result.err;
    workspace.size = 1;

    let mut tolerance = epsabs.max(epsrel * result.val.abs());

    if result.err <= 100.0 * f64::EPSILON * resabs && result.err > tolerance {
        result.code = IntegrationRetCode::RoundOffFirstIter;
        result.issue_warning(None);
        return result;
    } else if (result.err <= tolerance && result.err != resasc) || result.err == 0.0 {
        return result;
    } else if limit == 1 {
        result.code = IntegrationRetCode::OneIterNotEnough;
        result.issue_warning(None);
        return result;
    }

    // Initialization
    let mut table = ExtrapolationTable::new();
    table.append(result.val);

    let mut area = result.val;
    let mut errsum = result.err;
    let mut res_ext = result.val;
    let mut err_ext = f64::MAX;
    let positive_integrand = test_positivity(result.val, resabs);

    let mut extrapolate: bool = false;
    let mut disallow_extrapolation: bool = false;
    let mut roundoff_type1: usize = 0;
    let mut roundoff_type2: usize = 0;
    let mut roundoff_type3: usize = 0;

    let mut error_type2: bool = false;
    let mut error_over_large_intervals = 0.0;
    let mut ertest = 0.0;

    let mut ktmin: usize = 0;
    let mut correc = 0.0;

    let mut iter = 1;

    loop {
        // Bisect the subinterval with the largest error estimate
        let (a_i, b_i, r_i, e_i) = workspace.recieve();

        let current_level = workspace.level[workspace.i] + 1;

        let a1 = a_i;
        let b1 = 0.5 * (a_i + b_i);
        let a2 = b1;
        let b2 = b_i;

        iter += 1;

        let (area1, error1, _, resasc1) = fixed_order_gauss_kronrod(&f, a1, b1, key);
        let (area2, error2, _, resasc2) = fixed_order_gauss_kronrod(&f, a2, b2, key);

        let area12 = area1 + area2;
        let error12 = error1 + error2;
        let last_e_i = e_i;

        // Improve previous approximations to the integral and test for accuracy.
        errsum += error12 - e_i;
        area += area12 - r_i;

        tolerance = epsabs.max(epsrel * area.abs());

        if resasc1 != error1 && resasc2 != error2 {
            let delta = r_i - area12;

            if delta.abs() <= 1e-5 * area12.abs() && error12 >= 0.99 * e_i {
                if !extrapolate {
                    roundoff_type1 += 1;
                } else {
                    roundoff_type2 += 1;
                }
            }
            if iter > 10 && error12 > e_i {
                roundoff_type3 += 1;
            }
        }

        // Test for roundoff and eventually set error flag
        if roundoff_type1 + roundoff_type2 >= 10 || roundoff_type3 >= 20 {
            result.code = IntegrationRetCode::RoundOff;
        }
        if roundoff_type2 >= 5 {
            error_type2 = true;
        }

        // Set error flag in the case of bad integrand behavior at a point of the integraion range
        if subinterval_too_small(a1, a2, b2) {
            result.code = IntegrationRetCode::BadIntegrand;
        }

        // append the newly-created intervals to list
        workspace.update((a1, b1, area1, error1), (a2, b2, area2, error2));

        if errsum <= tolerance {
            result.val = workspace.sum_results();
            result.err = errsum;
            return result;
        }

        if result.code != IntegrationRetCode::Success {
            break;
        }

        if iter >= limit - 1 {
            result.code = IntegrationRetCode::TooManyIters;
            break;
        }

        if iter == 2 {
            // set up variables on first iter
            error_over_large_intervals = errsum;
            ertest = tolerance;
            table.append(area);
            continue;
        }

        if disallow_extrapolation {
            continue;
        }

        error_over_large_intervals -= last_e_i;

        if current_level < workspace.maximum_level {
            error_over_large_intervals += error12;
        }

        if !extrapolate {
            // test whether the interval to be bisected next is the smallest interval.
            if workspace.large_interval() {
                continue;
            }
            extrapolate = true;
            workspace.nrmax = 1;
        }
        if !error_type2 && error_over_large_intervals > ertest {
            if workspace.increase_nrmax() {
                continue;
            }
        }

        // Perform extrapolation
        table.append(area);

        let mut abseps = 0.0;
        let reseps = table.extrapolate(&mut abseps);

        ktmin += 1;

        if ktmin > 5 && err_ext < errsum * 1e-3 {
            result.code = IntegrationRetCode::DivergeSlowConverge;
        }
        if abseps < err_ext {
            ktmin = 0;
            err_ext = abseps;
            res_ext = reseps;
            correc = error_over_large_intervals;
            ertest = epsabs.max(epsrel * reseps.abs());
            if err_ext <= ertest {
                break;
            }
        }

        // Prepare bisection of the smallest interval.
        if table.n == 1 {
            disallow_extrapolation = true;
        }
        if result.code == IntegrationRetCode::DivergeSlowConverge {
            break;
        }

        // work on interval with largest error
        workspace.reset_nrmax();
        extrapolate = false;
        error_over_large_intervals = errsum;

        if iter >= limit {
            break;
        }
    }

    result.val = res_ext;
    result.err = err_ext;

    if err_ext == f64::MAX {
        result.val = workspace.sum_results();
        result.err = errsum;
        return result;
    }

    if result.code == IntegrationRetCode::TooManyIters || error_type2 {
        if error_type2 {
            err_ext = err_ext + correc;
        }
        if result.code == IntegrationRetCode::Success {
            result.code = IntegrationRetCode::BadIntegrand;
        }
        if res_ext != 0.0 && area != 0.0 {
            if err_ext * res_ext.abs().recip() > errsum * area.abs().recip() {
                result.val = workspace.sum_results();
                result.err = errsum;
                return result;
            }
        } else if err_ext > errsum {
            result.val = workspace.sum_results();
            result.err = errsum;
            return result;
        } else if area == 0.0 {
            return result;
        }
    }

    // Test on divergence
    {
        let max_area = res_ext.abs().max(area.abs());
        if !positive_integrand && max_area < 1e-2 * resabs {
            return result;
        }
    }

    {
        let ratio = res_ext * area.recip();

        if ratio < 1e-2 || ratio > 100.0 || errsum > area.abs() {
            result.code = IntegrationRetCode::Other;
        }
    }

    return result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_smooth() {
        let alpha = 2.6;
        let f = |x| f1(x, alpha);

        let exp_result = 7.716049382715789440E-02;
        let exp_abserr = 2.216394961010438404E-12;

        let result = qags(f, 0.0, 1.0, 0.0, 1e-10, 1000, 2);

        test_rel(result.val, exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-6);

        let result = qags(f, 1.0, 0.0, 0.0, 1e-10, 1000, 2);

        test_rel(result.val, -exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-6);
    }

    #[test]
    fn test_abs_bound() {
        let alpha = 2.0;
        let f = |x| f11(x, alpha);

        let exp_result = -5.908755278982136588E+03;
        let exp_abserr = 1.299646281053874554E-10;

        let result = qags(f, 1.0, 1000.0, 1e-7, 0.0, 1000, 2);

        test_rel(result.val, exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-3);

        let result = qags(f, 1000.0, 1.0, 1e-7, 0.0, 1000, 2);

        test_rel(result.val, -exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-3);
    }
}
