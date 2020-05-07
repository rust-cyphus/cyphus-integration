use num::Float;
use std::convert::From;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IntegrationRetcode {
    Success,
    OneIterNotEnough,
    RoundOffFirstIter,
    TooManyIters,
    RoundOff,
    BadIntegrand,
    RoundOffExtrapTable,
    DivergeSlowConverge,
    BadTol,
    Other,
}

impl Error for IntegrationRetcode {
    fn description(&self) -> &str {
        match *self {
            IntegrationRetcode::Success => "Successful integration",
            IntegrationRetcode::OneIterNotEnough => "A maximum of one iteration was insufficient",
            IntegrationRetcode::RoundOffFirstIter => {
                "cannot reach tolerance because of roundoff error on first attempt"
            }
            IntegrationRetcode::TooManyIters => "Number of iterations was insufficient",
            IntegrationRetcode::RoundOff => "Cannot reach tolerance because of roundoff error",
            IntegrationRetcode::BadIntegrand => {
                "Bad integrand behavior found in the integration interval"
            }
            IntegrationRetcode::RoundOffExtrapTable => {
                "Roundoff error detected in the extrapolation table"
            }
            IntegrationRetcode::DivergeSlowConverge => "Integral is divergent, or slowly convergent",
            IntegrationRetcode::BadTol => {
                "Invalid input tolerances. Tolerance cannot be achieved with given epsabs and epsrel"
            }
            IntegrationRetcode::Other => "Could not integrate function",
        }
    }
}

impl fmt::Display for IntegrationRetcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IntegrationRetcode::Success => write!(f,"Successful integration"),
            IntegrationRetcode::OneIterNotEnough => write!(f,"A maximum of one iteration was insufficient"),
            IntegrationRetcode::RoundOffFirstIter => {
                write!(f,"cannot reach tolerance because of roundoff error on first attempt")
            }
            IntegrationRetcode::TooManyIters => write!(f,"Number of iterations was insufficient"),
            IntegrationRetcode::RoundOff => write!(f,"Cannot reach tolerance because of roundoff error"),
            IntegrationRetcode::BadIntegrand => {
                write!(f,"Bad integrand behavior found in the integration interval")
            }
            IntegrationRetcode::RoundOffExtrapTable => {
                write!(f,"Roundoff error detected in the extrapolation table")
            }
            IntegrationRetcode::DivergeSlowConverge => write!(f,"Integral is divergent, or slowly convergent"),
            IntegrationRetcode::BadTol => {
                write!(f,"Invalid input tolerances. Tolerance cannot be achieved with given epsabs and epsrel")
            }
            IntegrationRetcode::Other => write!(f,"Could not integrate function"),
        }
    }
}

/// Given an integer error code, envoke the correct error
/// or return the result and error estimate.
pub fn handle_error<T: Float>(
    result: T,
    abserr: T,
    error_type: IntegrationRetcode,
) -> std::result::Result<(T, T), IntegrationRetcode> {
    match error_type {
        IntegrationRetcode::Success => Ok((result, abserr)),
        _ => Err::<(T, T), IntegrationRetcode>(error_type),
    }
}
