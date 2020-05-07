/// Workspace for adaptive integrators
pub struct IntegrationWorkSpace {
    pub size: usize,
    pub nrmax: usize,
    pub i: usize,
    pub maximum_level: usize,
    pub alist: Vec<f64>,
    pub blist: Vec<f64>,
    pub rlist: Vec<f64>,
    pub elist: Vec<f64>,
    pub order: Vec<usize>,
    pub level: Vec<usize>,
}

impl IntegrationWorkSpace {
    pub fn new(limit: usize) -> IntegrationWorkSpace {
        IntegrationWorkSpace {
            size: 0,
            nrmax: 0,
            i: 0,
            maximum_level: 0,
            alist: vec![0.0; limit],
            blist: vec![0.0; limit],
            rlist: vec![0.0; limit],
            elist: vec![0.0; limit],
            order: vec![0; limit],
            level: vec![0; limit],
        }
    }
    pub fn recieve(&self) -> (f64, f64, f64, f64) {
        let i = self.i;
        (self.alist[i], self.blist[i], self.rlist[i], self.elist[i])
    }
    pub fn update(&mut self, point1: (f64, f64, f64, f64), point2: (f64, f64, f64, f64)) {
        let (a1, b1, area1, error1) = point1;
        let (a2, b2, area2, error2) = point2;

        let imax = self.i;
        let inew = self.size;
        let newlevel = self.level[imax] + 1;

        if error2 > error1 {
            self.alist[imax] = a2;
            self.rlist[imax] = area2;
            self.elist[imax] = error2;
            self.level[imax] = newlevel;

            self.alist[inew] = a1;
            self.blist[inew] = b1;
            self.rlist[inew] = area1;
            self.elist[inew] = error1;
            self.level[inew] = newlevel;
        } else {
            self.blist[imax] = b1;
            self.rlist[imax] = area1;
            self.elist[imax] = error1;
            self.level[imax] = newlevel;

            self.alist[inew] = a2;
            self.blist[inew] = b2;
            self.rlist[inew] = area2;
            self.elist[inew] = error2;
            self.level[inew] = newlevel;
        }

        self.size += 1;
        if newlevel > self.maximum_level {
            self.maximum_level = newlevel;
        }
        self.sort();
    }
    pub fn large_interval(&self) -> bool {
        self.level[self.i] < self.maximum_level
    }
    pub fn increase_nrmax(&mut self) -> bool {
        let id = self.nrmax;
        let last = self.size - 1;
        let limit = self.alist.len();

        let jupbnd = if last > (1 + limit / 2) {
            limit + 1 - last
        } else {
            last
        };

        for _k in id..(jupbnd + 1) {
            let imax = self.order[self.nrmax];
            self.i = imax;
            if self.level[imax] < self.maximum_level {
                return true;
            }
            self.nrmax += 1;
        }
        false
    }
    pub fn reset_nrmax(&mut self) {
        self.nrmax = 0;
        self.i = self.order[0];
    }
    pub fn sum_results(&self) -> f64 {
        self.rlist
            .iter()
            .take(self.size)
            .fold(0.0, |acc, x| acc + *x)
    }
    pub fn sort(&mut self) {
        let last = self.size - 1;
        let limit = self.alist.len();

        let mut i_nrmax = self.nrmax;
        let mut i_maxerr = self.order[i_nrmax];

        // Check whether the list contains more than two error estimates
        if last < 2 {
            self.order[0] = 0;
            self.order[1] = 1;
            self.i = i_maxerr;
            return;
        }

        let mut errmax = self.elist[i_maxerr];

        // This part of the routine is only executed if, due to a difficult
        // integrand, subdivision increased the error estimate. In the normal
        // case the insert procedure should start after the nrmax-th largest
        // error estimate.
        while i_nrmax > 0 && errmax > self.elist[self.order[i_nrmax - 1]] {
            self.order[i_nrmax] = self.order[i_nrmax - 1];
            i_nrmax -= 1;
        }

        // Compute the number of elements in the list to be maintained in
        // descending order. This number depends on the number of subdivisions
        // still allowed.
        let top = if last < (limit / 2 + 2) {
            last
        } else {
            limit - last + 1
        };

        // Insert errmax by traversing the list top-down, starting comparison
        // from the element elist[order[i_nrmax+1]]
        let mut i = i_nrmax + 1;

        // The order of the tests in the following line is import to prevent
        // segmentation fault
        while i < top && errmax < self.elist[self.order[i]] {
            self.order[i - 1] = self.order[i];
            i += 1;
        }
        self.order[i - 1] = i_maxerr;

        // Insert errmin by traversing the list bottom-up
        let errmin = self.elist[last];
        let mut k = (top - 1) as i32;
        let ii = i as i32;

        while k > ii - 2 && errmin >= self.elist[self.order[k as usize]] {
            self.order[(k + 1) as usize] = self.order[k as usize];
            k -= 1;
        }
        let k = k as usize;

        self.order[k + 1] = last;

        // Set i_max and e_max
        i_maxerr = self.order[i_nrmax];
        self.i = i_maxerr;
        self.nrmax = i_nrmax;
    }
}

/// Workspace for QAWS integrator
struct IntegrationQawsTable<T> {
    alpha: T,
    beta: T,
    mu: i32,
    nu: i32,
    ri: Vec<T>,
    rj: Vec<T>,
    rg: Vec<T>,
    rh: Vec<T>,
}

enum IntegrationQawoEnum {
    Cosine,
    Sine,
}

/// Workspace for QAWO integrator
struct IntegrationQawoTable<T> {
    n: usize,
    omega: T,
    l: T,
    par: T,
    sine: IntegrationQawoEnum,
    chebmo: Vec<T>,
}
