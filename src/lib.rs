pub(crate) mod extrap;
pub mod qag;
pub mod qagi;
pub mod qagp;
pub mod qags;
pub(crate) mod qk;
pub mod result;
pub(crate) mod test_utils;
pub(crate) mod utils;
pub(crate) mod workspace;

use qagi::qagi;
use qagp::qagp;
use qags::qags;
use result::{IntegrationResult, IntegrationRetCode};

pub struct GaussKronrodIntegrator {
    pub epsabs: f64,
    pub epsrel: f64,
    pub limit: usize,
    pub key: u8,
    pub singular_points: Vec<f64>,
}

impl GaussKronrodIntegrator {
    pub fn integrate<F>(&self, f: F, a: f64, b: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if a == b {
            return IntegrationResult::new();
        }
        let (aa, bb) = if a < b { (a, b) } else { (b, a) };

        // [-inf, a, b, c, d, e, inf]
        // [-inf, a], [a, b, c, d, e], [e, inf]

        // Check if we have infinite endpoints
        if a.is_infinite() || b.is_infinite() {
            if !self.singular_points.is_empty() {
                let mut singular_points = self.singular_points.clone();
                singular_points.retain(|x| x.is_infinite());
                let res1 = qagp(
                    &f,
                    &singular_points,
                    self.epsabs,
                    self.epsrel,
                    self.limit,
                    self.key,
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
                        self.key,
                    );
                    res3 = qagi(
                        &f,
                        b,
                        *singular_points.last().unwrap(),
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        self.key,
                    );
                } else if a.is_infinite() {
                    res2 = qagi(
                        f,
                        a,
                        singular_points[0],
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        self.key,
                    );
                } else {
                    res3 = qagi(
                        f,
                        b,
                        *singular_points.last().unwrap(),
                        self.epsabs,
                        self.epsrel,
                        self.limit,
                        self.key,
                    );
                }
                return IntegrationResult {
                    val: res1.val + res2.val + res3.val,
                    err: res1.err + res2.err + res3.err,
                    code: IntegrationRetCode::Success,
                    nevals: res1.nevals + res2.nevals + res3.nevals,
                };
            } else {
                return qagi(f, a, b, self.epsabs, self.epsrel, self.limit, self.key);
            }
        } else if self.singular_points.is_empty() {
            return qags(f, a, b, self.epsabs, self.epsrel, self.limit, self.key);
        } else {
            let mut singular_points = vec![a];
            singular_points.reserve(self.singular_points.len() + 1);
            for pt in self.singular_points.iter() {
                singular_points.push(pt.clone());
            }
            singular_points.push(b);

            return qagp(
                f,
                &singular_points,
                self.epsabs,
                self.epsrel,
                self.limit,
                self.key,
            );
        }
    }
}

pub struct GaussKronrodIntegratorBuilder {
    pub epsabs: f64,
    pub epsrel: f64,
    pub limit: usize,
    pub key: u8,
    pub singular_points: Vec<f64>,
}

impl GaussKronrodIntegratorBuilder {
    pub fn default() -> Self {
        GaussKronrodIntegratorBuilder {
            epsabs: 1e-8,
            epsrel: 1e-8,
            limit: 1000,
            key: 2,
            singular_points: vec![],
        }
    }
    pub fn epsabs<'a>(&'a mut self, epsabs: f64) -> &'a mut Self {
        self.epsabs = epsabs;
        self
    }
    pub fn epsrel<'a>(&'a mut self, epsrel: f64) -> &'a mut Self {
        self.epsrel = epsrel;
        self
    }
    pub fn limit<'a>(&'a mut self, limit: usize) -> &'a mut Self {
        self.limit = limit;
        self
    }
    pub fn key<'a>(&'a mut self, key: u8) -> &'a mut Self {
        self.key = key;
        self
    }
    pub fn singular_points<'a>(&'a mut self, singular_points: Vec<f64>) -> &'a mut Self {
        self.singular_points = singular_points.clone();
        // Sort break-points and delete duplicates
        self.singular_points
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.singular_points.dedup_by(|a, b| (*a).eq(b));

        self
    }
    pub fn build(&self) -> GaussKronrodIntegrator {
        GaussKronrodIntegrator {
            epsabs: self.epsabs,
            epsrel: self.epsrel,
            limit: self.limit,
            key: self.key,
            singular_points: self.singular_points.clone(),
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
            .key(1)
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
            .key(1)
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
            .key(1)
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
            .epsabs(1e-7)
            .epsrel(0.0)
            .key(1)
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
            .key(1)
            .build();

        let f = |x: f64| (1.0 * x).exp();

        let result = gk.integrate(f, f64::NEG_INFINITY, 1.0);

        test_rel(result.val, exp_result, 1e-14);
        test_rel(result.err, exp_abserr, 1e-5);
    }
}
