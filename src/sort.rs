/// Maintains the descending ordering in the list of local
/// error estimates resulting from the interval subdivision
/// process. At each call, two error estimates are inserted
/// using the sequential search metho, top-down for the
/// largest error estimate and bottom-up for the smallest.
///
/// # Arguments
///
/// * `limit` - Maximum number of error estimates the list can contain.
/// * `last` - Number of error estimates currently in the list.
/// * `maxerr` - Index location of the nrmax-th largest error estimate currently in the list.
/// * `ermax` - `nrmax`-th  largest error estimate: ermax = elist[maxerr].
/// * `elist` - Array of dimension `last` containing the error estimates.
/// * `iord` - Array of dimension `last`. The first `k` elements contain index locations to the error estimates, such that `elist[iord[0]],...,elist[iord[k]]` form a decreasing sequence, with `k` = `last` if last <= (limit/2+2) and `k` = `limit + 1 - last` otherwise.
/// * `nrmax` - Index location of `maxerr`: `maxerr = iord[nrmax]`.
///
pub fn sort(
    limit: usize,
    last: usize,
    maxerr: &mut usize,
    ermax: &mut f64,
    elist: &mut [f64],
    iord: &mut [usize],
    nrmax: &mut usize,
) {
    if last > 1 {
        let errmax = elist[*maxerr];
        if *nrmax != 0 {
            let ido = *nrmax - 1;
            for _i in 0..(ido + 1) {
                let isucc = iord[*nrmax - 1];
                if errmax <= elist[isucc] {
                    break;
                }
                iord[*nrmax] = isucc;
                (*nrmax) -= 1;
            }
        }
        // 30
        let jupbn = if last > (limit / 2 + 2) {
            limit + 3 - last
        } else {
            last
        };
        let errmin = elist[last];
        let jbnd = jupbn - 1;
        let ibeg = *nrmax + 1;

        let mut goto50: bool = true;
        if ibeg <= jbnd {
            let mut i: usize = ibeg;
            for j in ibeg..(jbnd + 1) {
                let isucc = iord[j];
                if errmax >= elist[isucc] {
                    goto50 = false;
                    break;
                }
                iord[j - 1] = isucc;
                i += 1;
            }
            if !goto50 {
                // 60
                iord[i - 1] = *maxerr;
                let mut k = jbnd;
                let mut goto80: bool = false;
                for _j in i..(jbnd + 1) {
                    let isucc = iord[k];
                    if errmin < elist[isucc] {
                        goto80 = true;
                        break;
                    }
                    iord[k + 1] = isucc;
                    k -= 1;
                }
                if goto80 {
                    iord[k + 1] = last;
                } else {
                    iord[i] = last;
                }
            }
        }
        if goto50 {
            iord[jbnd] = *maxerr;
            iord[jupbn] = last;
        }
    } else {
        iord[0] = 0;
        iord[1] = 1;
    }
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
}
