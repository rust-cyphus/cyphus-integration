use crate::error::{handle_error, IntegrationRetcode};
use crate::extrap::ExtrapolationTable;
use crate::qk::fixed_order_gauss_kronrod;
use crate::utils::{subinterval_too_small, test_positivity};
use crate::workspace::IntegrationWorkSpace;
use num::Float;

#[allow(clippy::too_many_arguments)]
pub fn qags<F>(
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
    let mut result = 0.0;
    let mut abserr = 0.0;
    let mut workspace = IntegrationWorkSpace::new(limit);
    workspace.alist[0] = a;
    workspace.blist[0] = b;

    // Test on accuracy
    if epsabs <= 0.0 && (epsrel < 50.0 * f64::EPSILON || epsrel < f64::EPSILON) {
        return handle_error(result, abserr, IntegrationRetcode::BadTol);
    }

    // Perform the first integration
    let mut resabs: f64 = 0.0;
    let mut resasc: f64 = 0.0;

    result = fixed_order_gauss_kronrod(&f, a, b, key, &mut abserr, &mut resabs, &mut resasc);
    workspace.rlist[0] = result;
    workspace.elist[0] = abserr;
    workspace.size = 1;

    let mut tolerance = epsabs.max(epsrel * result.abs());

    if abserr <= 100.0 * f64::EPSILON * resabs && abserr > tolerance {
        return handle_error(result, abserr, IntegrationRetcode::RoundOffFirstIter);
    } else if (abserr <= tolerance && abserr != resasc) || abserr == 0.0 {
        return Ok((result, abserr));
    } else if limit == 1 {
        return handle_error(result, abserr, IntegrationRetcode::OneIterNotEnough);
    }

    // Initialization
    let mut table = ExtrapolationTable::new();
    table.append(result);

    let mut area = result;
    let mut errsum = abserr;
    let mut res_ext = result;
    let mut err_ext = f64::MAX;
    let positive_integrand = test_positivity(result, resabs);

    let mut extrapolate: bool = false;
    let mut disallow_extrapolation: bool = false;
    let mut roundoff_type1: usize = 0;
    let mut roundoff_type2: usize = 0;
    let mut roundoff_type3: usize = 0;

    let mut error_type: IntegrationRetcode = IntegrationRetcode::Success;
    let mut error_type2: bool = false;
    let mut error_over_large_intervals = 0.0;
    let mut ertest = 0.0;

    let mut ktmin: usize = 0;
    let mut correc = 0.0;

    for iter in 1..limit {
        // Bisect the subinterval with the largest error estimate
        let (a_i, b_i, r_i, e_i) = workspace.recieve();

        let current_level = workspace.level[workspace.i] + 1;

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
            error_type = IntegrationRetcode::RoundOff; // Roundoff error
        }
        if roundoff_type2 >= 5 {
            error_type2 = true;
        }

        // Set error flag in the case of bad integrand behavior at a point of the integraion range
        if subinterval_too_small(a1, a2, b2) {
            error_type = IntegrationRetcode::BadIntegrand;
        }

        // append the newly-created intervals to list
        workspace.update((a1, b1, area1, error1), (a2, b2, area2, error2));

        if errsum <= tolerance {
            result = workspace.sum_results();
            abserr = errsum;
            return handle_error(result, abserr, error_type);
        }

        if error_type != IntegrationRetcode::Success {
            break;
        }

        if iter >= limit - 1 {
            error_type = IntegrationRetcode::TooManyIters;
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

        error_over_large_intervals = error_over_large_intervals - last_e_i;

        if current_level < workspace.maximum_level {
            error_over_large_intervals = error_over_large_intervals + error12;
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
            error_type = IntegrationRetcode::DivergeSlowConverge;
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
        if error_type == IntegrationRetcode::DivergeSlowConverge {
            break;
        }

        // work on interval with largest error
        workspace.reset_nrmax();
        extrapolate = false;
        error_over_large_intervals = errsum;
    }

    result = res_ext;
    abserr = err_ext;

    if (err_ext - f64::MAX).abs() > f64::EPSILON {
        result = workspace.sum_results();
        abserr = errsum;
        return handle_error(result, abserr, error_type);
    }

    if error_type == IntegrationRetcode::TooManyIters || error_type2 {
        if error_type2 {
            err_ext = err_ext + correc;
        }
        if error_type == IntegrationRetcode::Success {
            error_type = IntegrationRetcode::BadIntegrand;
        }
        if res_ext.abs() > f64::EPSILON && area.abs() > f64::EPSILON {
            if err_ext * res_ext.abs().recip() > errsum * area.abs().recip() {
                result = workspace.sum_results();
                abserr = errsum;
                return handle_error(result, abserr, error_type);
            }
        } else if err_ext > errsum {
            result = workspace.sum_results();
            abserr = errsum;
            return handle_error(result, abserr, error_type);
        } else if area.abs() < f64::EPSILON {
            return handle_error(result, abserr, error_type);
        }
    }

    // Test on divergence
    {
        let max_area = res_ext.abs().max(area.abs());
        if !positive_integrand && max_area < 1e-2 * resabs {
            return handle_error(result, abserr, error_type);
        }
    }

    {
        let ratio = res_ext * area.recip();

        if ratio < 1e-2 || ratio > 100.0 || errsum > area.abs() {
            error_type = IntegrationRetcode::Other;
        }
    }

    return handle_error(result, abserr, error_type);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_rel;

    fn f1(x: f64, alpha: f64) -> f64 {
        x.powf(alpha) * x.recip().ln()
    }
    fn f11(x: f64, alpha: f64) -> f64 {
        x.recip().ln().powf(alpha - 1.0)
    }

    #[test]
    fn test_smooth() {
        let alpha = 2.6;
        let f = |x| f1(x, alpha);

        let exp_result = 7.716049382715789440E-02;
        let exp_abserr = 2.216394961010438404E-12;

        let result = qags(f, 0.0, 1.0, 0.0, 1e-10, 1000, 2);

        match result {
            Ok(result) => {
                test_rel(result.0, exp_result, 1e-15);
                test_rel(result.1, exp_abserr, 1e-6);
            }
            Err(err) => panic!("{:?}", err),
        }

        let result = qags(f, 1.0, 0.0, 0.0, 1e-10, 1000, 2);

        match result {
            Ok(result) => {
                test_rel(result.0, -exp_result, 1e-15);
                test_rel(result.1, exp_abserr, 1e-6);
            }
            Err(err) => panic!("{:?}", err),
        }
    }
    #[test]
    fn test_abs_bound() {
        let alpha = 2.0;
        let f = |x| f11(x, alpha);

        let exp_result = -5.908755278982136588E+03;
        let exp_abserr = 1.299646281053874554E-10;

        let result = qags(f, 1.0, 1000.0, 1e-7, 0.0, 1000, 2);

        dbg!(result);

        match result {
            Ok(result) => {
                test_rel(result.0, exp_result, 1e-15);
                test_rel(result.1, exp_abserr, 1e-3);
            }
            Err(err) => panic!("{:?}", err),
        }

        let result = qags(f, 1000.0, 1.0, 1e-7, 0.0, 1000, 2);

        match result {
            Ok(result) => {
                test_rel(result.0, -exp_result, 1e-15);
                test_rel(result.1, exp_abserr, 1e-3);
            }
            Err(err) => panic!("{:?}", err),
        }
    }
}
