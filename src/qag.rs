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
    key: i8,
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
    if epsabs <= 0.0 && epsrel < 50.0 * f64::EPSILON {
        return handle_error(result, abserr, IntegrationRetcode::BadTol);
    }

    // Perform the first integration
    let mut resabs = 0.0;
    let mut resasc = 0.0;
    result = fixed_order_gauss_kronrod(&f, a, b, key, &mut abserr, &mut resabs, &mut resasc);
    workspace.rlist[0] = result;
    workspace.elist[0] = abserr;

    // Test on accuracy
    let mut tolerance = epsabs.max(epsrel * result.abs());
    // need IEEE rounding here to match original quadpack behavior
    // NOTE: GSL uses a volitile variable for extended precision registers. Should we do the same?
    let round_off = 50.0 * f64::EPSILON * resabs;

    if abserr <= round_off && abserr > tolerance {
        return handle_error(result, abserr, IntegrationRetcode::RoundOffFirstIter);
    } else if (abserr <= tolerance && (abserr - resasc).abs() > f64::EPSILON)
        || abserr.abs() < f64::EPSILON
    {
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

        errsum = errsum + (error12 - e_i);
        area = area + area12 - r_i;

        if (resabs1 - error1).abs() > f64::EPSILON && (resasc2 - error2).abs() > f64::EPSILON {
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

        if iter >= limit || error_type != IntegrationRetcode::Success || errsum <= tolerance {
            break;
        }

        iter += 1;
    }

    result = workspace.sum_results();
    abserr = errsum;

    if errsum <= tolerance {
        error_type = IntegrationRetcode::Success;
    } else if iter == limit {
        error_type = IntegrationRetcode::TooManyIters;
    }

    return handle_error(result, abserr, error_type);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gauss() {}
}
