/// Data structure for performing the epsilon extrapolation algorithm
pub(crate) struct ExtrapolationTable {
    /// rlist2[n] contains the new element in the first column of the epsilon
    /// table
    pub n: usize,
    /// containing the elements of the two lower diagonals of the triangular
    /// epsilon table. the elements are numbered starting at the right-hand
    /// corner of the triangle.
    pub rlist2: [f64; 52],
    /// Number of calls to the extrapolate method
    pub nres: usize,
    /// Contains the last 3 results
    pub res3la: [f64; 3],
}

impl ExtrapolationTable {
    /// Construct an empty extrapolation table filled with zeros.
    pub(crate) fn new() -> ExtrapolationTable {
        ExtrapolationTable {
            n: 0,
            rlist2: [0.0; 52],
            nres: 0,
            res3la: [0.0; 3],
        }
    }
    /// Add new results to the extrapolation table
    pub(crate) fn append(&mut self, y: f64) {
        self.rlist2[self.n] = y;
        self.n += 1;
    }
    /// Determines the limit of a given sequence of approximations, by means of
    /// the epsilon algorithm of p.wynn. an estimate of the absolute error is
    /// also given. The condensed epsilon table is computed. only those elements
    /// needed for the computation of the next diagonal are preserved.
    pub(crate) fn extrapolate(&mut self, abserr: &mut f64) -> f64 {
        let n = self.n - 1;

        let current = self.rlist2[n];
        let mut absolute = f64::MAX;
        let mut relative = 5.0 * current.abs();

        let newelm = n / 2;
        let norig = n;
        let mut nfinal = n;
        let nres_orig = self.nres;

        let mut result = current;
        *abserr = f64::MAX;

        if n < 2 {
            result = current;
            *abserr = absolute.max(relative);
            return result;
        }
        self.rlist2[n + 2] = self.rlist2[n];
        self.rlist2[n] = f64::MAX;

        for i in 0..newelm {
            let mut res = self.rlist2[n - 2 * i + 2];
            let e0 = self.rlist2[n - 2 * i - 2];
            let e1 = self.rlist2[n - 2 * i - 1];
            let e2 = res;

            let e1abs = e1.abs();
            let delta2 = e2 - e1;
            let err2 = delta2.abs();
            let tol2 = e2.abs().max(e1abs) * f64::EPSILON;
            let delta3 = e1 - e0;
            let err3 = delta3.abs();
            let tol3 = e1abs.max(e0.abs()) * f64::EPSILON;

            if err2 <= tol2 && err3 <= tol3 {
                // If e0, e1 and e2 are equal to within machine accuracy,
                // convergence is assumed.
                result = res;
                absolute = err2 + err3;
                relative = 5.0 * f64::EPSILON * res.abs();
                *abserr = absolute.max(relative);
                return result;
            }

            let e3 = self.rlist2[n - 2 * i];
            self.rlist2[n - 2 * i] = e1;
            let delta1 = e1 - e3;
            let err1 = delta1.abs();
            let tol1 = e1abs.max(e3.abs()) * f64::EPSILON;

            // If two elements are very close to each other, omit a part of
            // the table by adjusting the value of n
            if err1 <= tol1 || err2 <= tol2 || err3 <= tol3 {
                nfinal = 2 * i;
                break;
            }

            let ss = delta1.recip() + delta2.recip() - delta3.recip();

            // Test to detect irregular behaviour in the table, and eventually
            // omit a part of the table by adjusting the value of n.
            if (ss * e1).abs() <= 1e-4 {
                nfinal = 2 * i;
                break;
            }

            // Compute a new element and eventually adjust the value of result.
            res = e1 + ss.recip();
            self.rlist2[n - 2 * i] = res;

            {
                let error = err2 + (res - e2).abs() + err3;
                if error <= *abserr {
                    *abserr = error;
                    result = res;
                }
            }
        }

        // Shift the table
        {
            let limexp: usize = 50 - 1;
            if nfinal == limexp {
                nfinal = 2 * (limexp / 2);
            }
        }

        if norig % 2 == 1 {
            for i in 0..(newelm + 1) {
                self.rlist2[1 + i * 2] = self.rlist2[i * 2 + 3]
            }
        } else {
            for i in 0..(newelm + 1) {
                self.rlist2[i * 2] = self.rlist2[i * 2 + 2];
            }
        }

        if norig != nfinal {
            for i in 0..(nfinal + 1) {
                self.rlist2[i] = self.rlist2[norig - nfinal + i];
            }
        }

        self.n = nfinal + 1;

        if nres_orig < 3 {
            self.res3la[nres_orig] = result;
            *abserr = f64::MAX;
        } else {
            // Compute error estimate
            *abserr = (result - self.res3la[2]).abs()
                + (result - self.res3la[1]).abs()
                + (result - self.res3la[0]).abs();

            self.res3la[0] = self.res3la[1];
            self.res3la[1] = self.res3la[2];
            self.res3la[2] = result;
        }

        self.nres = nres_orig + 1;

        *abserr = (*abserr).max(5.0 * f64::EPSILON * result.abs());

        result
    }
}
