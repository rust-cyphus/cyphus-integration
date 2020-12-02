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

use crate::qags::qags;
use crate::result::IntegrationResult;
use num::Float;

/// Integrate a function over an infinite, semi-infinit or finite interval by
/// transforming the integrand to a new function defined over a finite interval
/// (if nesseccary) and using `qags`.
pub fn qagi<T, F>(
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
    let sign = if a < b { T::one() } else { -T::one() };
    let (aa, bb) = if a < b { (a, b) } else { (b, a) };

    if bb >= T::infinity() && aa <= T::neg_infinity() {
        // Integrate from -inf to inf by transforming the integrand using:
        // f(t) -> [f( (1 - t) / t ) + f( -(1 - t) / t)] / t^2
        let transformed = |t: T| -> T {
            let x = (T::one() - t) / t;
            sign * (f(x) + f(-x)) / (t * t)
        };
        qags(
            transformed,
            T::zero(),
            T::one(),
            epsabs,
            epsrel,
            limit,
            nodes,
            kronrod,
            gauss,
        )
    } else if bb >= T::infinity() && aa > T::neg_infinity() {
        // Integrate from a to inf by transforming the integrand using:
        // f(t) -> f( a + (1 - t) / t ) / t^2
        let transformed = |t: T| -> T {
            let x = aa + (T::one() - t) / t;
            sign * f(x) / (t * t)
        };
        qags(
            transformed,
            T::zero(),
            T::one(),
            epsabs,
            epsrel,
            limit,
            nodes,
            kronrod,
            gauss,
        )
    } else if aa <= T::neg_infinity() && bb < T::infinity() {
        // Integrate from -inf to a by transforming the integrand using:
        // f(t) -> f( a + (1 - t) / t ) / t^2
        let transformed = |t: T| -> T {
            let x = bb - (T::one() - t) * t.recip();
            sign * f(x) * t.powi(2).recip()
        };
        qags(
            transformed,
            T::zero(),
            T::one(),
            epsabs,
            epsrel,
            limit,
            nodes,
            kronrod,
            gauss,
        )
    } else {
        let transformed = |t: T| -> T {
            let x = t * bb + aa * (T::one() - t);
            sign * f(x) * (bb - aa)
        };
        qags(
            transformed,
            T::zero(),
            T::one(),
            epsabs,
            epsrel,
            limit,
            nodes,
            kronrod,
            gauss,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qk::QK15;
    use crate::test_utils::*;

    #[test]
    fn test_0_inf_f455() {
        let exp_result = -3.616892186127022568E-01;
        let exp_abserr = 3.016716913328831851E-06;

        let result = qagi(
            f455,
            0.0,
            f64::INFINITY,
            0.0,
            1e-3,
            1000,
            &QK15.0,
            &QK15.1,
            &QK15.2,
        );

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_0_inf_f15() {
        let exp_result = 6.553600000000024738E+04;
        let exp_abserr = 7.121667111456009280E-04;

        let alpha = 5.0;
        let f = |x| f15(x, alpha);
        let result = qagi(
            f,
            0.0,
            f64::INFINITY,
            0.0,
            1e-7,
            1000,
            &QK15.0,
            &QK15.1,
            &QK15.2,
        );

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_0_inf_f16() {
        let exp_result = 1.000000000006713292E-04;
        let exp_abserr = 3.084062020905636316E-09;

        let alpha = 1.0;
        let f = |x| f16(x, alpha);
        let result = qagi(
            f,
            99.9,
            f64::INFINITY,
            1e-7,
            0.0,
            1000,
            &QK15.0,
            &QK15.1,
            &QK15.2,
        );

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_inf_inf() {
        let exp_result = 2.275875794468747770E+00;
        let exp_abserr = 7.436490118267390744E-09;

        let alpha = 1.0;
        let f = |x: f64| (-x - x * x).exp();

        let result = qagi(
            f,
            f64::NEG_INFINITY,
            f64::INFINITY,
            1e-7,
            0.0,
            1000,
            &QK15.0,
            &QK15.1,
            &QK15.2,
        );

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_inf_0() {
        let exp_result = 2.718281828459044647E+00;
        let exp_abserr = 1.588185109253204805E-10;

        let alpha = 1.0;
        let f = |x: f64| (alpha * x).exp();

        let result = qagi(
            f,
            f64::NEG_INFINITY,
            1.0,
            1e-7,
            0.0,
            1000,
            &QK15.0,
            &QK15.1,
            &QK15.2,
        );

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }
}
