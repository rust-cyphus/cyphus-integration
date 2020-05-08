use log::warn;

/// Codes for the result of an integration
#[derive(Debug, PartialEq, Eq)]
pub enum IntegrationRetCode {
    /// Successful integration
    Success,
    /// When limit == 1 and tolerance can't be met
    OneIterNotEnough,
    /// Round-off error encountered during first iteration
    RoundOffFirstIter,
    /// More iterations are required to reach tolerance
    TooManyIters,
    /// Round-off error encountered during integration
    RoundOff,
    /// Singularity encountered
    BadIntegrand,
    /// Round-off error encountered in extrapolation table
    RoundOffExtrapTable,
    /// Integral is either divergence or slowly convergent
    DivergeSlowConverge,
    /// Invalid tolerances passed to integrator
    BadTol,
    /// Invalid argument passed to integrator
    InvalidArg,
    /// Something else when wrong
    Other,
}

/// Structure for the result of an integration
#[derive(Debug)]
pub struct IntegrationResult {
    /// Value of the integration
    pub val: f64,
    /// Estimated error of the integration
    pub err: f64,
    /// Return code
    pub code: IntegrationRetCode,
}

impl IntegrationResult {
    pub(crate) fn issue_warning(&self, args: Option<&[f64]>) {
        match self.code {
            IntegrationRetCode::Success => {}
            IntegrationRetCode::OneIterNotEnough => warn!("A maximum of one iteration was insufficient"),
            IntegrationRetCode::RoundOffFirstIter => {
                warn!("cannot reach tolerance because of round-off error on first attempt")
            }
            IntegrationRetCode::TooManyIters => warn!("Number of iterations was insufficient"),
            IntegrationRetCode::RoundOff => warn!("Cannot reach tolerance because of round-off
             error"),
            IntegrationRetCode::BadIntegrand => {
                warn!("Bad integrand behavior found in the integration interval")
            }
            IntegrationRetCode::RoundOffExtrapTable => {
                warn!("Round-off error detected in the extrapolation table")
            }
            IntegrationRetCode::DivergeSlowConverge => warn!("Integral is divergent, or slowly convergent"),
            IntegrationRetCode::BadTol => {
                warn!("Invalid input tolerances. Tolerance cannot be achieved with given \
                epsabs and epsrel: {:?}", args)
            }
            IntegrationRetCode::InvalidArg => {
                warn!("Invalid argument(s). Args = {:?}", args)
            }
            IntegrationRetCode::Other => warn!("Could not integrate function"),
        }
    }
}