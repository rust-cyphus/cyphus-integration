// GSL License:
//
// Copyright (C) 1996, 1997, 1998, 1999, 2000, 2001, 2007 Brian Gough
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or (at
// your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

use crate::extrap::ExtrapolationTable;
use crate::qk::qk;
use crate::result::{IntegrationResult, IntegrationRetCode};
use crate::utils::{subinterval_too_small, test_positivity};
use crate::workspace::IntegrationWorkSpace;
use num::Float;

pub fn qags<T, F>(
    f: F,
    a: T,
    b: T,
    epsabs: T,
    epsrel: T,
    limit: usize,
    nodes: &[T],
    kronrod: &[T],
    gauss: &[T],
) -> IntegrationResult<T>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut result = IntegrationResult::<T>::new();

    let mut workspace = IntegrationWorkSpace::new(limit);
    workspace.alist[0] = a;
    workspace.blist[0] = b;

    // Test on accuracy
    if epsabs <= T::zero()
        && (epsrel < T::from(50).unwrap() * T::epsilon() || epsrel < T::epsilon())
    {
        result.code = IntegrationRetCode::BadTol;
        result.issue_warning(Some(&[epsabs, epsrel]));
        return result;
    }

    // Perform the first integration
    let (val, err, resabs, resasc) = qk(&f, a, b, nodes, kronrod, gauss);
    result.val = val;
    result.err = err;

    workspace.rlist[0] = result.val;
    workspace.elist[0] = result.err;
    workspace.size = 1;

    let mut tolerance = epsabs.max(epsrel * result.val.abs());

    if result.err <= T::from(100).unwrap() * T::epsilon() * resabs && result.err > tolerance {
        result.code = IntegrationRetCode::RoundOffFirstIter;
        result.issue_warning(None);
        return result;
    } else if (result.err <= tolerance && result.err != resasc) || result.err.is_zero() {
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
    let mut err_ext = T::max_value();
    let positive_integrand = test_positivity(result.val, resabs);

    let mut extrapolate: bool = false;
    let mut disallow_extrapolation: bool = false;
    let mut roundoff_type1: usize = 0;
    let mut roundoff_type2: usize = 0;
    let mut roundoff_type3: usize = 0;

    let mut error_type2: bool = false;
    let mut error_over_large_intervals = T::zero();
    let mut ertest = T::zero();

    let mut ktmin: usize = 0;
    let mut correc = T::zero();

    let mut iter = 1;

    loop {
        // Bisect the subinterval with the largest error estimate
        let (a_i, b_i, r_i, e_i) = workspace.recieve();

        let current_level = workspace.level[workspace.i] + 1;

        let a1 = a_i;
        let b1 = T::from(0.5).unwrap() * (a_i + b_i);
        let a2 = b1;
        let b2 = b_i;

        iter += 1;

        let (area1, error1, _, resasc1) = qk(&f, a1, b1, nodes, kronrod, gauss);
        let (area2, error2, _, resasc2) = qk(&f, a2, b2, nodes, kronrod, gauss);

        let area12 = area1 + area2;
        let error12 = error1 + error2;
        let last_e_i = e_i;

        // Improve previous approximations to the integral and test for accuracy.
        errsum = errsum + error12 - e_i;
        area = area + area12 - r_i;

        tolerance = epsabs.max(epsrel * area.abs());

        if resasc1 != error1 && resasc2 != error2 {
            let delta = r_i - area12;

            if delta.abs() <= T::from(1e-5).unwrap() * area12.abs()
                && error12 >= T::from(0.99).unwrap() * e_i
            {
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
        if !error_type2 && error_over_large_intervals > ertest && workspace.increase_nrmax() {
            continue;
        }

        // Perform extrapolation
        table.append(area);

        let mut abseps = T::zero();
        let reseps = table.extrapolate(&mut abseps);

        ktmin += 1;

        if ktmin > 5 && err_ext < errsum * T::from(1e-3).unwrap() {
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

    if err_ext == T::max_value() {
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
        if !res_ext.is_zero() && !area.is_zero() {
            if err_ext * res_ext.abs().recip() > errsum * area.abs().recip() {
                result.val = workspace.sum_results();
                result.err = errsum;
                return result;
            }
        } else if err_ext > errsum {
            result.val = workspace.sum_results();
            result.err = errsum;
            return result;
        } else if area.is_zero() {
            return result;
        }
    }

    // Test on divergence
    {
        let max_area = res_ext.abs().max(area.abs());
        if !positive_integrand && max_area < T::from(1e-2).unwrap() * resabs {
            return result;
        }
    }

    {
        let ratio = res_ext * area.recip();

        if ratio < T::from(1e-2).unwrap() || ratio > T::from(100).unwrap() || errsum > area.abs() {
            result.code = IntegrationRetCode::Other;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qk::QK21;
    use crate::test_utils::*;

    #[test]
    fn test_smooth() {
        let alpha = 2.6;
        let f = |x| f1(x, alpha);

        let exp_result = 7.716049382715789440E-02;
        let exp_abserr = 2.216394961010438404E-12;

        let result = qags(f, 0.0, 1.0, 0.0, 1e-10, 1000, &QK21.0, &QK21.1, &QK21.2);

        test_rel(result.val, exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-6);

        let result = qags(f, 1.0, 0.0, 0.0, 1e-10, 1000, &QK21.0, &QK21.1, &QK21.2);

        test_rel(result.val, -exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-6);
    }

    #[test]
    fn test_abs_bound() {
        let alpha = 2.0;
        let f = |x| f11(x, alpha);

        let exp_result = -5.908755278982136588E+03;
        let exp_abserr = 1.299646281053874554E-10;

        let result = qags(f, 1.0, 1000.0, 1e-7, 0.0, 1000, &QK21.0, &QK21.1, &QK21.2);

        test_rel(result.val, exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-3);

        let result = qags(f, 1000.0, 1.0, 1e-7, 0.0, 1000, &QK21.0, &QK21.1, &QK21.2);

        test_rel(result.val, -exp_result, 1e-15);
        test_rel(result.err, exp_abserr, 1e-3);
    }
}
