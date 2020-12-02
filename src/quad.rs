use crate::gausskronrod::kronrod;
use crate::qagi::qagi;
use crate::qagp::qagp;
use crate::qags::qags;
use crate::result::{IntegrationResult, IntegrationRetCode};
use num::Float;

#[derive(Clone, Debug)]
/// One-dimensional Gauss-Kronrod integrator.
pub struct GaussKronrodIntegrator<T: Float> {
    /// Absolute tolerance.
    pub epsabs: T,
    /// Relative tolerance.
    pub epsrel: T,
    /// Total number of allowed refinements.
    pub limit: usize,
    /// Locations of the singular points of the integrand.
    pub singular_points: Vec<T>,
    /// Gauss-Kronrod nodes
    nodes: Vec<T>,
    /// Kronrod weights
    kronrod: Vec<T>,
    /// Gauss weights
    gauss: Vec<T>,
}

impl<T: Float> GaussKronrodIntegrator<T> {
    /// Integrate a function `f` over the interval `a` to `b`, returning
    /// an `IntegrationResult` object. For infinite bounds, use
    /// f64::INFINITY.
    ///
    /// # Examples
    ///
    /// Integrate a function over a finite interval: f(x) = x^2
    /// ```
    /// let gk = GaussKronrodIntegratorBuilder::default()
    ///     .reltol(1e-8)
    ///     .abstol(1e-8)
    ///     .build();
    /// let f = |x:f64| x * x;
    /// let res = gk.integrate(f, 0.0, 1.0);
    /// assert!((res.val - 1.0 / 3.0).abs() < 1e-8);
    /// ```
    ///
    /// Integrate a function over an infinite interval: f(x) = exp(-x^2)/sqrt(2pi)
    /// ```
    /// let gk = GaussKronrodIntegratorBuilder::default()
    ///     .reltol(1e-8)
    ///     .abstol(1e-8)
    ///     .build();
    /// let f = |x:f64| (-x * x / 2.0).exp() / (2.0 * std::f64::const::PI).sqrt();
    /// let res = gk.integrate(f, f64::NEG_INFINITY, f64::INFINITY);
    /// assert!((res.val - 1.0).abs() < 1e-8);
    /// ```
    pub fn integrate<F>(&self, f: F, a: T, b: T) -> IntegrationResult<T>
    where
        F: Fn(T) -> T,
    {
        // Check if we have infinite endpoints
        if a.is_infinite() || b.is_infinite() {
            if !self.singular_points.is_empty() {
                let mut singular_points = self.singular_points.clone();
                singular_points.retain(|x| x.is_finite());
                let res1 = qagp(
                    &f,
                    &singular_points,
                    self.epsabs,
                    self.epsrel,
                    self.limit,
                    &self.nodes,
                    &self.kronrod,
                    &self.gauss,
                );
                let mut res2 = IntegrationResult::new();
                let mut res3 = IntegrationResult::new();
                if a.is_infinite() && b.is_infinite() {
                    res2 = qagi(
                        &f,
                        a,
                        singular_points[0],
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        &self.nodes,
                        &self.kronrod,
                        &self.gauss,
                    );
                    res3 = qagi(
                        &f,
                        *singular_points.last().unwrap(),
                        b,
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        &self.nodes,
                        &self.kronrod,
                        &self.gauss,
                    );
                } else if a.is_infinite() {
                    res2 = qagi(
                        f,
                        a,
                        singular_points[0],
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        &self.nodes,
                        &self.kronrod,
                        &self.gauss,
                    );
                } else {
                    res3 = qagi(
                        f,
                        *singular_points.last().unwrap(),
                        b,
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        &self.nodes,
                        &self.kronrod,
                        &self.gauss,
                    );
                }
                IntegrationResult {
                    val: res1.val + res2.val + res3.val,
                    err: res1.err + res2.err + res3.err,
                    code: IntegrationRetCode::Success,
                    nevals: res1.nevals + res2.nevals + res3.nevals,
                }
            } else {
                qagi(
                    f,
                    a,
                    b,
                    self.epsabs,
                    self.epsrel,
                    self.limit,
                    &self.nodes,
                    &self.kronrod,
                    &self.gauss,
                )
            }
        } else if self.singular_points.is_empty() {
            qags(
                f,
                a,
                b,
                self.epsabs,
                self.epsrel,
                self.limit,
                &self.nodes,
                &self.kronrod,
                &self.gauss,
            )
        } else {
            let mut singular_points = vec![a];
            singular_points.reserve(self.singular_points.len() + 1);
            for pt in self.singular_points.iter() {
                singular_points.push(*pt);
            }
            singular_points.push(b);

            qagp(
                f,
                &singular_points,
                self.epsabs,
                self.epsrel,
                self.limit,
                &self.nodes,
                &self.kronrod,
                &self.gauss,
            )
        }
    }
}

/// Builder struct used to construct an integrator with wanted parameters.
pub struct GaussKronrodIntegratorBuilder<T: Float> {
    /// Absolute tolerance.
    epsabs: Option<T>,
    /// Relative tolerance.
    epsrel: Option<T>,
    /// Total number of allowed refinements.
    limit: Option<usize>,
    /// Locations of the singular points of the integrand.
    singular_points: Option<Vec<T>>,
    /// Order of gauss-kronrod rule
    order: Option<usize>,
}

impl<T: Float> GaussKronrodIntegratorBuilder<T> {
    pub fn default() -> Self {
        GaussKronrodIntegratorBuilder {
            epsabs: None,
            epsrel: None,
            limit: None,
            singular_points: None,
            order: None,
        }
    }
    /// Set the absolute tolerance.
    pub fn epsabs(mut self, epsabs: T) -> Self {
        self.epsabs = Some(epsabs);
        self
    }
    /// Set the relative tolerance.
    pub fn epsrel(mut self, epsrel: T) -> Self {
        self.epsrel = Some(epsrel);
        self
    }
    /// Set the total number of allowed refinements.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    /// Set the order of the gauss-kronrod rule
    pub fn order(mut self, order: usize) -> Self {
        self.order = Some(order);
        self
    }
    /// Specify the locations of the singular points of the integrand.
    pub fn singular_points(mut self, singular_points: Vec<T>) -> Self {
        let mut sp = singular_points;
        // Sort break-points and delete duplicates
        sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sp.dedup_by(|a, b| (*a).eq(b));
        self.singular_points = Some(sp);

        self
    }
    /// Build the integrator.
    pub fn build(self) -> GaussKronrodIntegrator<T> {
        let order = self.order.unwrap_or(7);

        let (ns, ks, gs) = match order {
            7 => (
                crate::qk::QK15
                    .0
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK15
                    .1
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK15
                    .2
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
            ),
            10 => (
                crate::qk::QK21
                    .0
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK21
                    .1
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK21
                    .2
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
            ),
            15 => (
                crate::qk::QK31
                    .0
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK31
                    .1
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK31
                    .2
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
            ),
            20 => (
                crate::qk::QK41
                    .0
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK41
                    .1
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK41
                    .2
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
            ),
            25 => (
                crate::qk::QK51
                    .0
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK51
                    .1
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK51
                    .2
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
            ),
            30 => (
                crate::qk::QK61
                    .0
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK61
                    .1
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
                crate::qk::QK61
                    .2
                    .to_vec()
                    .iter()
                    .map(|x| T::from(*x).unwrap())
                    .collect::<Vec<T>>(),
            ),
            _ => kronrod(order),
        };

        let sp = self.singular_points.unwrap_or(vec![]);
        GaussKronrodIntegrator {
            epsabs: self.epsabs.unwrap_or(T::from(1e-8).unwrap()),
            epsrel: self.epsrel.unwrap_or(T::from(1e-8).unwrap()),
            limit: self.limit.unwrap_or(1000),
            singular_points: sp,
            nodes: ns,
            kronrod: ks,
            gauss: gs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_0_inf_f455() {
        let exp_result = -3.616892186127022568E-01;
        let exp_abserr = 3.016716913328831851E-06;

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(0.0)
            .epsrel(1e-3)
            .order(7)
            .build();
        let result = gk.integrate(f455, 0.0, f64::INFINITY);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_0_inf_f15() {
        let exp_result = 6.553600000000024738E+04;
        let exp_abserr = 7.121667111456009280E-04;

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(0.0)
            .epsrel(1e-7)
            .order(7)
            .build();

        let f = |x| f15(x, 5.0);
        let result = gk.integrate(f, 0.0, f64::INFINITY);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_0_inf_f16() {
        let exp_result = 1.000000000006713292E-04;
        let exp_abserr = 3.084062020905636316E-09;

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(1e-7)
            .epsrel(0.0)
            .order(7)
            .build();

        let alpha = 1.0;
        let f = |x| f16(x, alpha);
        let result = gk.integrate(f, 99.9, f64::INFINITY);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_inf_inf() {
        let exp_result = 2.275875794468747770E+00;
        let exp_abserr = 7.436490118267390744E-09;

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(1e-3)
            .epsrel(0.0)
            .order(7)
            .build();

        let f = |x: f64| (-x - x * x).exp();

        let result = gk.integrate(f, f64::NEG_INFINITY, f64::INFINITY);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_inf_0() {
        let exp_result = 2.718281828459044647E+00;
        let exp_abserr = 1.588185109253204805E-10;

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(1e-7)
            .epsrel(0.0)
            .order(7)
            .build();

        let f = |x: f64| (1.0 * x).exp();

        let result = gk.integrate(f, f64::NEG_INFINITY, 1.0);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_454() {
        let f = |x| f454(x);

        let exp_result = 5.274080611672716401E+01;
        let exp_abserr = 1.755703848687062418E-04;
        let points = vec![1.0, 2f64.sqrt()];

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(0.0)
            .epsrel(1e-3)
            .singular_points(points)
            .order(10)
            .limit(1000)
            .build();
        let result = gk.integrate(f, 0.0, 3.0);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }

    #[test]
    fn test_gaussian() {
        let f = |x: f64| (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();

        let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(1e-8)
            .epsrel(1e-8)
            .build();
        let result = gk.integrate(f, f64::NEG_INFINITY, f64::INFINITY);
        println!("{:?}", result);
        assert!((result.val - 1.0).abs() < 1e-8);
    }
}
