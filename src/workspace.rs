use num::Float;

/// Workspace for adaptive integrators
pub struct IntegrationWorkSpace<T> {
    pub size: usize,
    pub nrmax: usize,
    pub i: usize,
    pub maximum_level: usize,
    pub alist: Vec<T>,
    pub blist: Vec<T>,
    pub rlist: Vec<T>,
    pub elist: Vec<T>,
    pub order: Vec<usize>,
    pub level: Vec<usize>,
}

impl<T: Float> IntegrationWorkSpace<T> {
    pub fn new(limit: usize) -> IntegrationWorkSpace<T> {
        IntegrationWorkSpace {
            size: 0,
            nrmax: 0,
            i: 0,
            maximum_level: 0,
            alist: vec![T::zero(); limit],
            blist: vec![T::zero(); limit],
            rlist: vec![T::zero(); limit],
            elist: vec![T::zero(); limit],
            order: vec![0; limit],
            level: vec![0; limit],
        }
    }
    pub fn recieve(&self) -> (T, T, T, T) {
        let i = self.i;
        (self.alist[i], self.blist[i], self.rlist[i], self.elist[i])
    }
    pub fn update(&mut self, point1: (T, T, T, T), point2: (T, T, T, T)) {
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
    pub fn sum_results(&self) -> T {
        let mut sum = T::zero();
        for val in self.rlist.iter().take(self.size) {
            sum = sum + *val;
        }
        sum
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
