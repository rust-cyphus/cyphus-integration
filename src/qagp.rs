//! qagp
//!
//! Implements the adaptive Gauss-Kronrod integration for integrals with known
//! singular points. Adapted from the GNU Scientific Library. See:
//! Brian Gough. 2009. GNU Scientific Library Reference Manual - Third Edition (3rd. ed.). Network Theory Ltd.
//!
//! # Examples
//! Integrate f(x) = x^3 ln(|(x^2-1)(x^2-2)|) from 0 to 3. We can see that this
//! function is singular at x = 1 and x = sqrt(2).
//! ```
//! let f = |x: f64| ((x * x - 1.0) * (x * x - 2.0)).abs().ln() * x.powi(3);
//! let pts = vec![0.0, 1.0, 2.0f64.sqrt(), 3.0];
//! let result = qagp(f, &pts, 1e-3, 0.0, 1000, 2);
//! let analytic = 52.7407483834712;
//! assert!((result.val - analytic).abs() < 1e-3);
//! ```

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
use crate::qk::fixed_order_gauss_kronrod;
use crate::result::*;
use crate::utils::{subinterval_too_small, test_positivity};
use crate::workspace::*;

/// Integrate the function `f` over an interval with known singularities. The
/// singularities are specified in the `points` vector, which should include
/// the endpoints of integration.
///
/// # Arguments
/// - `f`: Function to integrate.
/// - `points`: A vector specifying endpoints of the integration range as well
///             as the singular points.
/// - `epsabs`: Requested absolute tolerance.
/// - `epsrel`: Requested relative tolerance.
/// - `limit`: Maximum number of subdivisions allowed.
/// - `key`: Specifies which quadrature rule to use (i.e. key=1 => 15pt, key=2 => 21, key=3 => 31,...,key >= 6 => 61)
///
/// # Examples
/// Integrate x^3 Ln|(x^2-1)(x^2-2)| over (0, 3). There are singularities at
/// x = 1 and x = sqrt(2).
/// ```
/// let f = |x: f64| ((x * x - 1.0) * (x * x - 2.0)).abs().ln() * x.powi(3);
/// let pts = vec![0.0, 1.0, 2.0f64.sqrt(), 3.0];
/// let result = qagp(f, &pts, 1e-3, 0.0, 1000, 2);
/// let analytic = 52.7407483834712;
/// assert!((result.val - analytic).abs() < 1e-3);
/// ```
pub(crate) fn qagp<F>(
    f: F,
    points: &Vec<f64>,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    key: u8,
) -> IntegrationResult
where
    F: Fn(f64) -> f64,
{
    let mut result = IntegrationResult::new();
    let mut resabs0: f64 = 0.0;

    // Enforce that the break-points are in ascending order
    let mut pts = (*points).clone();
    // Sort break-points and delete duplicates
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup_by(|a, b| (*a).eq(b));

    // Number of intervals
    let nint = pts.len() - 1;

    let mut workspace = IntegrationWorkSpace::new(limit);
    workspace.alist[0] = 0.0;
    workspace.blist[0] = 0.0;

    // Initialize results
    if points.len() > limit {
        result.code = IntegrationRetCode::InvalidArg;
        result.issue_warning(Some(pts.as_ref()));
        return result;
    }
    if epsabs <= 0.0 && (epsrel < 50.0 * f64::EPSILON || epsrel < 0.5e-28) {
        result.code = IntegrationRetCode::BadTol;
        result.issue_warning(Some(&[epsabs, epsrel]));
        return result;
    }

    // Perform the first integration
    for i in 0..nint {
        let a1 = pts[i];
        let b1 = pts[i + 1];

        let (val, err, resabs, resasc) = fixed_order_gauss_kronrod(&f, a1, b1, key);

        result.val += val;
        result.err += err;
        resabs0 += resabs;

        workspace.append_interval(a1, b1, val, err);
        workspace.level[i] = if err == resasc && err != 0.0 { 1 } else { 0 };
    }

    // Compute the initial error estimate
    let mut errsum = 0.0;
    for i in 0..nint {
        if workspace.level[i] != 0 {
            workspace.elist[i] = result.err;
        }
        errsum += workspace.elist[i];
    }

    for val in workspace.level.iter_mut().take(nint) {
        *val = 0;
    }

    // Sort results into order of decreasing error via the indirection
    // array order
    workspace.sort_results();

    // Test on accuracy
    let mut tolerance = epsabs.max(epsrel * result.val.abs());

    if result.err <= 100.0 * f64::EPSILON * resabs0 && result.err > tolerance {
        result.code = IntegrationRetCode::RoundOffFirstIter;
        result.issue_warning(None);
        return result;
    } else if result.err <= tolerance {
        result.code = IntegrationRetCode::Success;
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
    let mut res_ext = result.val;
    let mut err_ext = f64::MAX;
    let mut error_over_large_intervals = errsum;
    let mut ertest = tolerance;
    let positive_integrand = test_positivity(result.val, resabs0);

    // Main loop
    let mut extrapolate = false;
    let mut disallow_extrapolation = false;
    let mut roundoff_type1 = 0;
    let mut roundoff_type2 = 0;
    let mut roundoff_type3 = 0;
    let mut error_type2 = false;
    let mut correc = 0.0;
    let mut ktmin = 0;
    let mut iter = nint - 1;

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
        let last_ei = e_i;

        // Improve previous approximations to the integral and test for
        // accuracy.
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

        // set error flag in the case of bad integrand behaviour at
        // a point of the integration range
        if subinterval_too_small(a1, a2, b2) {
            result.code = IntegrationRetCode::BadIntegrand;
        }

        // Append the newly-created intervals to the list
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
        if disallow_extrapolation {
            continue;
        }

        error_over_large_intervals -= last_ei;

        if current_level < workspace.maximum_level {
            error_over_large_intervals += error12;
        }

        if !extrapolate {
            // Test whether the interval to be bisected next is the smallest
            // interval
            if workspace.large_interval() {
                continue;
            }

            extrapolate = true;
            workspace.nrmax = 1;
        }

        // The smallest interval has the largest error.  Before
        // bisecting decrease the sum of the errors over the larger
        // intervals (error_over_large_intervals) and perform
        // extrapolation.
        if !error_type2 && error_over_large_intervals > ertest {
            if workspace.increase_nrmax() {
                continue;
            }
        }

        // Perform extrapolation
        table.append(area);

        if table.n >= 3 {
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

            // Prepare bisection of the smallest interval
            if table.n == 1 {
                disallow_extrapolation = true;
            }
            if result.code == IntegrationRetCode::DivergeSlowConverge {
                break;
            }
        }

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
            err_ext += correc;
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
    let max_area = res_ext.abs().max(area.abs());
    if !positive_integrand && max_area < 1e-2 * resabs0 {
        return result;
    }
    let ratio = res_ext * area.recip();
    if ratio < 1e-2 || ratio > 1e2 || errsum > area.abs() {
        result.code = IntegrationRetCode::Other;
    }
    return result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test() {
        let f = |x| f454(x);

        let exp_result = 5.274080611672716401E+01;
        let exp_abserr = 1.755703848687062418E-04;
        let points = vec![0.0, 1.0, 2f64.sqrt(), 3.0];

        let result = qagp(f, &points, 0.0, 1e-3, 1000, 2);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }
    #[test]
    fn doctest() {
        let f = |x: f64| ((x * x - 1.0) * (x * x - 2.0)).abs().ln() * x.powi(3);
        let pts = vec![0.0, 1.0, 2.0f64.sqrt(), 3.0];
        let result = dbg!(qagp(f, &pts, 1e-3, 0.0, 1000, 2));
        let analytic = 52.7407483834712;
        assert!((result.val - analytic).abs() < 1e-3);
    }
}
